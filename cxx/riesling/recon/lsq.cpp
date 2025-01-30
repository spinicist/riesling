#include "inputs.hpp"
#include "outputs.hpp"

#include "rl/algo/lsmr.hpp"
#include "rl/log.hpp"
#include "rl/op/pad.hpp"
#include "rl/op/recon.hpp"
#include "rl/precon.hpp"
#include "rl/scaling.hpp"
#include "rl/sense/sense.hpp"
#include "rl/types.hpp"

using namespace rl;

void main_recon_lsq(args::Subparser &parser)
{
  CoreArgs               coreArgs(parser);
  GridArgs<3>            gridArgs(parser);
  PreconArgs             preArgs(parser);
  ReconArgs              reconArgs(parser);
  SENSEArgs              senseArgs(parser);
  LSMRArgs               lsqArgs(parser);
  f0Args                 f0Args(parser);
  ArrayFlag<float, 3>    cropFov(parser, "FOV", "Crop FoV in mm (x,y,z)", {"crop-fov"}, Eigen::Array3f::Zero());
  args::ValueFlag<Index> debugIters(parser, "I", "Write debug images ever N iterations (1)", {"debug-iters"}, 1);

  ParseCommand(parser, coreArgs.iname, coreArgs.oname);
  auto const  cmd = parser.GetCommand().Name();
  HD5::Reader reader(coreArgs.iname.Get());
  Info const  info = reader.readInfo();
  Trajectory  traj(reader, info.voxel_size, coreArgs.matrix.Get());
  auto        noncart = reader.readTensor<Cx5>();
  traj.checkDims(FirstN<3>(noncart.dimensions()));

  auto const basis = LoadBasis(coreArgs.basisFile.Get());
  auto const R =
    Recon(reconArgs.Get(), preArgs.Get(), gridArgs.Get(), senseArgs.Get(), traj, basis.get(), f0Args.Get(), noncart);
  Log::Debug(cmd, "A {} {} M {}", R.A->ishape, R.A->oshape, R.M->ishape);
  auto debug = [shape = R.A->ishape, d = debugIters.Get()](Index const i, LSMR::Vector const &x) {
    if (i % d == 0) { Log::Tensor(fmt::format("lsmr-x-{:02d}", i), shape, x.data(), HD5::Dims::Images); }
  };
  LSMR lsmr{R.A, R.M, nullptr, lsqArgs.Get(), debug};

  auto const x = lsmr.run(CollapseToConstVector(noncart));
  auto const xm = AsTensorMap(x, R.A->ishape);

  TOps::Pad<Cx, 5> oc(traj.matrixForFOV(cropFov.Get(), R.A->ishape[3], R.A->ishape[4]), R.A->ishape);
  auto             out = oc.adjoint(xm);
  if (basis) { basis->applyR(out); }
  WriteOutput(cmd, coreArgs.oname.Get(), out, HD5::Dims::Images, info);
  if (coreArgs.residual) {
    WriteResidual(cmd, coreArgs.oname.Get(), reconArgs.Get(), gridArgs.Get(), senseArgs.Get(), preArgs.Get(), traj, xm, R.A,
                  noncart);
  }
  Log::Print(cmd, "Finished");
}
