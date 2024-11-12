#include "types.hpp"

#include "algo/lsmr.hpp"
#include "inputs.hpp"
#include "io/hd5.hpp"
#include "log.hpp"
#include "op/fft.hpp"
#include "op/ndft.hpp"
#include "op/nufft.hpp"
#include "precon.hpp"

using namespace rl;
using namespace std::literals::complex_literals;

void main_psf(args::Subparser &parser)
{
  CoreArgs    coreArgs(parser);
  GridArgs<3> gridArgs(parser);
  PreconArgs  preArgs(parser);
  LsqOpts     lsqOpts(parser);

  args::Flag mtf(parser, "M", "Save Modulation Transfer Function", {"mtf"});

  ArrayFlag<float, 2> phases(parser, "P", "Phase accrued at start and end of spoke", {"phases", 'p'});

  ParseCommand(parser, coreArgs.iname, coreArgs.oname);
  auto const  cmd = parser.GetCommand().Name();
  HD5::Reader reader(coreArgs.iname.Get());
  Trajectory  traj(reader, reader.readInfo().voxel_size, coreArgs.matrix.Get());
  auto const  basis = LoadBasis(coreArgs.basisFile.Get());
  Index const nB = basis->nB();
  auto const  A = TOps::NUFFT<3>::Make(gridArgs.Get(), traj, 1, basis.get());
  auto const  M = MakeKspacePre(preArgs.Get(), gridArgs.Get(), traj, 1, 1, 1, basis.get());
  LSMR const  lsmr{A, M, nullptr, lsqOpts.its.Get(), lsqOpts.atol.Get(), lsqOpts.btol.Get(), lsqOpts.ctol.Get()};

  float const startPhase = phases.Get()[0];
  float const endPhase = phases.Get()[1];

  Eigen::VectorXcf const trace = Eigen::ArrayXcf::LinSpaced(traj.nSamples(), startPhase * 1if, endPhase * 1if).exp() *
                                 std::sqrt(nB / (float)Product(LastN<3>(A->ishape)));
  Eigen::TensorMap<Cx1 const> traceM(trace.data(), Sz1{traj.nSamples()});

  Cx3         ks = traceM.reshape(Sz3{1, traj.nSamples(), 1}).broadcast(Sz3{1, 1, traj.nTraces()});
  auto        x = lsmr.run(CollapseToConstVector(ks));
  auto        xm = AsTensorMap(x, LastN<4>(A->ishape));
  HD5::Writer writer(coreArgs.oname.Get());
  writer.writeTensor(HD5::Keys::Data, xm.dimensions(), xm.data(), {"v", "i", "j", "k"});

  if (mtf) {
    auto const fft = TOps::FFT<4, 3>(xm.dimensions());
    Log::Print(cmd, "Calculating MTF");
    xm *= xm.constant(std::sqrt(Product(LastN<3>(xm.dimensions()))));
    fft.forward(xm);
    writer.writeTensor("mtf", xm.dimensions(), xm.data(), {"v", "i", "j", "k"});
  }
  Log::Print(cmd, "Finished");
}
