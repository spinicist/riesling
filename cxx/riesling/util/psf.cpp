#include "inputs.hpp"

#include "rl/algo/lsmr.hpp"
#include "rl/io/hd5.hpp"
#include "rl/log/log.hpp"
#include "rl/op/compose.hpp"
#include "rl/op/fft.hpp"
#include "rl/op/ndft.hpp"
#include "rl/op/nufft.hpp"
#include "rl/op/reshape.hpp"
#include "rl/op/sense.hpp"
#include "rl/precon.hpp"
#include "rl/types.hpp"

using namespace rl;
using namespace std::literals::complex_literals;

void main_psf(args::Subparser &parser)
{
  CoreArgs<3>         coreArgs(parser);
  GridArgs<3>         gridArgs(parser);
  PreconArgs          preArgs(parser);
  SENSEArgs<3>        senseArgs(parser);
  LSMRArgs            lsqOpts(parser);
  ArrayFlag<float, 3> cropFov(parser, "FOV", "Crop FoV in mm (x,y)", {"crop-fov"});

  args::Flag tpsf(parser, "T", "Output Transform PSF", {'t', "tpsf"});
  ParseCommand(parser, coreArgs.iname, coreArgs.oname);
  auto const           cmd = parser.GetCommand().Name();
  HD5::Reader          input(coreArgs.iname.Get());
  Trajectory           traj(input, input.readStruct<Info>(HD5::Keys::Info).voxel_size, coreArgs.matrix.Get());
  auto const           basis = LoadBasis(coreArgs.basisFile.Get());
  auto const           gridOpts = gridArgs.Get();
  TOps::TOp<4, 3>::Ptr A = nullptr;
  if (senseArgs.type) {
    SENSE::Opts senseOpts = senseArgs.Get();
    HD5::Reader senseReader(senseOpts.type);
    Cx5 const   skern = senseReader.readTensor<Cx5>(HD5::Keys::Data);
    auto        sense =
      TOps::MakeSENSE(SENSE::KernelsToMaps<3>(skern, traj.matrixForFOV(gridOpts.fov), gridOpts.osamp), basis ? basis->nB() : 1);
    auto nufft = TOps::MakeNUFFT<3>(gridOpts, traj, sense->oshape[4], basis.get());
    A = TOps::MakeCompose(sense, nufft);
  } else {
    auto nufft = TOps::MakeNUFFT<3>(gridOpts, traj, 1, basis.get());
    A = TOps::MakeReshapeInput(nufft, FirstN<4>(nufft->ishape));
  }

  auto const   shape = A->ishape;
  Index const  nC = A->oshape[0];
  auto const   M = MakeKSpacePrecon(preArgs.Get(), gridOpts, traj, basis.get(), nC, Sz0{});
  TOps::Pad<4> C(Concatenate(traj.matrixForFOV(cropFov.Get()), LastN<1>(shape)), shape);
  LSMR const   lsmr{A, M, nullptr, lsqOpts.Get()};
  Cx4          imgs(shape);
  imgs.setZero();
  Cx3 ks(nC, traj.nSamples(), traj.nTraces());

  if (tpsf) {
    Index const nB = shape[3];
    Cx5         output(AddBack(FirstN<3>(C.ishape), nB, nB));
    for (Index ib = 0; ib < nB; ib++) {
      imgs.setZero();
      imgs(shape[0] / 2, shape[1] / 2, shape[2] / 2, ib) = 1.f;
      ks = A->forward(imgs);
      auto x = lsmr.run(CollapseToConstVector(ks));
      output.chip<4>(ib) = C.adjoint(AsTensorMap(x, shape));
    }
    HD5::Writer writer(coreArgs.oname.Get());
    writer.writeStruct(HD5::Keys::Info, input.readStruct<Info>(HD5::Keys::Info));
    writer.writeTensor(HD5::Keys::Data, output.dimensions(), output.data(), {"i", "j", "k", "b1", "b2"});
  } else {
    imgs.chip<2>(shape[2] / 2).chip<1>(shape[1] / 2).chip<0>(shape[0] / 2).setConstant(1.f);
    ks = A->forward(imgs);
    auto const  x = lsmr.run(CollapseToConstVector(ks));
    Cx4 const   out = C.adjoint(AsTensorMap(x, shape));
    HD5::Writer writer(coreArgs.oname.Get());
    writer.writeStruct(HD5::Keys::Info, input.readStruct<Info>(HD5::Keys::Info));
    writer.writeTensor(HD5::Keys::Data, out.dimensions(), out.data(), {"i", "j", "k", "b"});
  }

  Log::Print(cmd, "Finished");
}
