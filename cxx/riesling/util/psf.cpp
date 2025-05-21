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
  LSMRArgs     lsqOpts(parser);

  ArrayFlag<float, 2> phases(parser, "P", "Phase accrued at start and end of spoke", {"phases", 'p'});

  ParseCommand(parser, coreArgs.iname, coreArgs.oname);
  auto const  cmd = parser.GetCommand().Name();
  HD5::Reader input(coreArgs.iname.Get());
  Trajectory  traj(input, input.readInfo().voxel_size, coreArgs.matrix.Get());
  auto const  basis = LoadBasis(coreArgs.basisFile.Get());
  Index const nB = basis ? basis->nB() : 1;
  auto const  A = TOps::NUFFT<3>::Make(gridArgs.Get(), traj, 1, basis.get());
  auto const  M = MakeKSpacePrecon(preArgs.Get(), gridArgs.Get(), traj, 1, Sz0{});
  LSMR const  lsmr{A, M, nullptr, lsqOpts.Get()};

  float const startPhase = phases.Get()[0];
  float const endPhase = phases.Get()[1];

  Eigen::VectorXcf const trace = Eigen::ArrayXcf::LinSpaced(traj.nSamples(), startPhase * 1if, endPhase * 1if).exp() *
                                 std::sqrt(nB / (float)Product(LastN<3>(A->ishape)));
  Eigen::TensorMap<Cx1 const> traceM(trace.data(), Sz1{traj.nSamples()});

  Cx3         ks = traceM.reshape(Sz3{1, traj.nSamples(), 1}).broadcast(Sz3{1, 1, traj.nTraces()});
  auto        x = lsmr.run(CollapseToConstVector(ks));
  HD5::Writer writer(coreArgs.oname.Get());
  writer.writeInfo(input.readInfo());
  writer.writeTensor(HD5::Keys::Data, FirstN<4>(A->ishape), x.data(), {"i", "j", "k", "b"});
  Log::Print(cmd, "Finished");
}
