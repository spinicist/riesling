#include "inputs.hpp"

#include "rl/algo/lsmr.hpp"
#include "rl/io/hd5.hpp"
#include "rl/log/log.hpp"
#include "rl/op/fft.hpp"
#include "rl/op/ndft.hpp"
#include "rl/op/nufft.hpp"
#include "rl/precon.hpp"
#include "rl/types.hpp"

using namespace rl;
using namespace std::literals::complex_literals;

void main_psf(args::Subparser &parser)
{
  CoreArgs<3>         coreArgs(parser);
  GridArgs<3>         gridArgs(parser);
  PreconArgs          preArgs(parser);
  LSMRArgs            lsqOpts(parser);
  ArrayFlag<float, 3> cropFov(parser, "FOV", "Crop FoV in mm (x,y)", {"crop-fov"});

  args::Flag tpsf(parser, "T", "Output Transform PSF", {'t', "tpsf"});
  ParseCommand(parser, coreArgs.iname, coreArgs.oname);
  auto const   cmd = parser.GetCommand().Name();
  HD5::Reader  input(coreArgs.iname.Get());
  Trajectory   traj(input, input.readStruct<Info>(HD5::Keys::Info).voxel_size, coreArgs.matrix.Get());
  auto const   basis = LoadBasis(coreArgs.basisFile.Get());
  auto const   A = TOps::MakeNUFFT<3>(gridArgs.Get(), traj, 1, basis.get());
  auto const   M = MakeKSpacePrecon(preArgs.Get(), gridArgs.Get(), traj, 1, Sz0{});
  auto const   shape = A->ishape;
  TOps::Pad<5> C(Concatenate(traj.matrixForFOV(cropFov.Get()), LastN<2>(shape)), shape);
  LSMR const   lsmr{A, M, nullptr, lsqOpts.Get()};

  if (tpsf) {
    Index const nB = shape[4];
    Cx5         imgs(shape);
    Cx5         output(AddBack(FirstN<3>(C.ishape), nB, nB));

    for (Index ib = 0; ib < nB; ib++) {
      imgs.setZero();
      imgs(shape[0] / 2, shape[1] / 2, shape[2] / 2, 0, ib) = 1.f;
      Cx3 const ks = A->forward(imgs);
      auto      x = lsmr.run(CollapseToConstVector(ks));
      output.chip<4>(ib) = C.adjoint(AsTensorMap(x, shape)).chip<3>(0); // Get rid of channel dimension
    }
    HD5::Writer writer(coreArgs.oname.Get());
    writer.writeStruct(HD5::Keys::Info, input.readStruct<Info>(HD5::Keys::Info));
    writer.writeTensor(HD5::Keys::Data, output.dimensions(), output.data(), {"i", "j", "k", "b1", "b2"});
  } else {
    Cx3 ks(1, traj.nSamples(), traj.nTraces());
    ks.setConstant(1.f);
    auto const  x = lsmr.run(CollapseToConstVector(ks));
    Cx4 const   out = C.adjoint(AsTensorMap(x, shape)).chip<3>(0);
    HD5::Writer writer(coreArgs.oname.Get());
    writer.writeStruct(HD5::Keys::Info, input.readStruct<Info>(HD5::Keys::Info));
    writer.writeTensor(HD5::Keys::Data, out.dimensions(), out.data(), {"i", "j", "k", "b"});
  }

  Log::Print(cmd, "Finished");
}
