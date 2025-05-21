#include "inputs.hpp"

#include "rl/algo/lsmr.hpp"
#include "rl/io/hd5.hpp"
#include "rl/log.hpp"
#include "rl/op/fft.hpp"
#include "rl/op/ndft.hpp"
#include "rl/op/nufft.hpp"
#include "rl/precon.hpp"
#include "rl/types.hpp"

using namespace rl;
using namespace std::literals::complex_literals;

void main_psf(args::Subparser &parser)
{
  CoreArgs<3> coreArgs(parser);
  GridArgs<3> gridArgs(parser);
  PreconArgs  preArgs(parser);
  LSMRArgs    lsqOpts(parser);

  args::Flag tpsf(parser, "T", "Output Transform PSF", {'t', "tpsf"});
  ParseCommand(parser, coreArgs.iname, coreArgs.oname);
  auto const  cmd = parser.GetCommand().Name();
  HD5::Reader input(coreArgs.iname.Get());
  Trajectory  traj(input, input.readInfo().voxel_size, coreArgs.matrix.Get());
  auto const  basis = LoadBasis(coreArgs.basisFile.Get());
  Index const nB = basis ? basis->nB() : 1;
  auto const  A = TOps::NUFFT<3>::Make(gridArgs.Get(), traj, 1, basis.get());
  auto const  M = MakeKSpacePrecon(preArgs.Get(), gridArgs.Get(), traj, 1, Sz0{});
  LSMR const  lsmr{A, M, nullptr, lsqOpts.Get()};

  if (tpsf) {
    auto const  shape = A->ishape;
    Index const nB = shape[4];
    Cx5         imgs(shape);
    Cx5         output(AddBack(FirstN<3>(shape), nB, nB));

    for (Index ib = 0; ib < nB; ib++) {
      imgs.setZero();
      imgs(shape[0] / 2, shape[1] / 2, shape[2] / 2, 0, ib) = 1.f;
      Cx3 const ks = A->forward(imgs);
      auto      x = lsmr.run(CollapseToConstVector(ks));
      output.chip<4>(ib) = AsTensorMap(x, shape).chip<3>(0); // Get rid of channel dimension
    }
    HD5::Writer writer(coreArgs.oname.Get());
    writer.writeInfo(input.readInfo());
    writer.writeTensor(HD5::Keys::Data, output.dimensions(), output.data(), {"i", "j", "k", "b1", "b2"});
  } else {
    Cx3 ks(1, traj.nSamples(), traj.nTraces());
    ks.setConstant(1.f);
    auto        x = lsmr.run(CollapseToConstVector(ks));
    HD5::Writer writer(coreArgs.oname.Get());
    writer.writeInfo(input.readInfo());
    writer.writeTensor(HD5::Keys::Data, FirstN<4>(A->ishape), x.data(), {"i", "j", "k", "b"});
  }

  Log::Print(cmd, "Finished");
}
