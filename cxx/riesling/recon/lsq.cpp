#include "types.hpp"

#include "algo/lsmr.hpp"
#include "inputs.hpp"
#include "log.hpp"
#include "op/recon.hpp"
#include "outputs.hpp"
#include "precon.hpp"
#include "scaling.hpp"
#include "sense/sense.hpp"

using namespace rl;

void main_recon_lsq(args::Subparser &parser)
{
  CoreArgs               coreArgs(parser);
  GridArgs<3>            gridArgs(parser);
  PreconArgs             preArgs(parser);
  ReconArgs              reconArgs(parser);
  SENSE::Opts            senseOpts(parser);
  LsqOpts                lsqOpts(parser);
  ArrayFlag<float, 3>    cropFov(parser, "FOV", "Crop FoV in mm (x,y,z)", {"crop-fov"}, Eigen::Array3f::Zero());
  args::ValueFlag<Index> debugIters(parser, "I", "Write debug images ever N iterations (1)", {"debug-iters"}, 1);
  ParseCommand(parser, coreArgs.iname, coreArgs.oname);
  auto const  cmd = parser.GetCommand().Name();
  HD5::Reader reader(coreArgs.iname.Get());
  Info const  info = reader.readInfo();
  Trajectory  traj(reader, info.voxel_size, coreArgs.matrix.Get());
  auto        noncart = reader.readTensor<Cx5>();
  traj.checkDims(FirstN<3>(noncart.dimensions()));
  Index const nC = noncart.dimension(0);
  Index const nS = noncart.dimension(3);
  Index const nT = noncart.dimension(4);

  auto const basis = LoadBasis(coreArgs.basisFile.Get());
  auto const A = Recon::Choose(reconArgs.Get(), gridArgs.Get(), senseOpts, traj, basis.get(), noncart);
  auto const M = MakeKspacePre(preArgs.Get(), gridArgs.Get(), traj, nC, nS, nT, basis.get());
  Log::Debug(cmd, "A {} {} M {} {}", A->ishape, A->oshape, M->rows(), M->cols());
  auto debug = [shape = A->ishape, d = debugIters.Get()](Index const i, LSMR::Vector const &x) {
    if (i % d == 0) { Log::Tensor(fmt::format("lsmr-x-{:02d}", i), shape, x.data(), HD5::Dims::Image); }
  };
  LSMR lsmr{A, M, nullptr, lsqOpts.its.Get(), lsqOpts.atol.Get(), lsqOpts.btol.Get(), lsqOpts.ctol.Get(), debug};

  auto const x = lsmr.run(CollapseToConstVector(noncart), lsqOpts.λ.Get());
  auto const xm = AsTensorMap(x, A->ishape);

  TOps::Pad<Cx, 5> oc(traj.matrixForFOV(cropFov.Get(), A->ishape[0], nT), A->ishape);
  auto             out = oc.adjoint(xm);
  if (basis) { basis->applyR(out); }
  WriteOutput(cmd, coreArgs.oname.Get(), out, HD5::Dims::Image, info);
  if (coreArgs.residual) {
    WriteResidual(cmd, coreArgs.oname.Get(), reconArgs.Get(), gridArgs.Get(), senseOpts, preArgs.Get(), traj, xm, A, noncart);
  }
  Log::Print(cmd, "Finished");
}
