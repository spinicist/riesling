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
  CoreOpts    coreOpts(parser);
  GridOpts    gridOpts(parser);
  PreconOpts  preOpts(parser);
  SENSE::Opts senseOpts(parser);
  LsqOpts     lsqOpts(parser);
  Array3fFlag cropFov(parser, "FOV", "Crop FoV in mm (x,y,z)", {"crop-fov"}, Eigen::Array3f::Zero());

  ParseCommand(parser, coreOpts.iname, coreOpts.oname);
  auto const  cmd = parser.GetCommand().Name();
  HD5::Reader reader(coreOpts.iname.Get());
  Info const  info = reader.readInfo();
  Trajectory  traj(reader, info.voxel_size, gridOpts.matrix.Get());
  auto        noncart = reader.readTensor<Cx5>();
  traj.checkDims(FirstN<3>(noncart.dimensions()));
  Index const nC = noncart.dimension(0);
  Index const nS = noncart.dimension(3);
  Index const nT = noncart.dimension(4);

  auto const basis = LoadBasis(coreOpts.basisFile.Get());
  auto const A = Recon::Choose(gridOpts, senseOpts, traj, basis.get(), noncart);
  auto const M = MakeKspacePre(traj, nC, nS, nT, basis.get(), preOpts.type.Get(), preOpts.bias.Get());
  Log::Debug(cmd, "A {} {} M {} {}", A->ishape, A->oshape, M->rows(), M->cols());
  auto debug = [shape = A->ishape](Index const i, LSMR::Vector const &x) {
    Log::Tensor(fmt::format("lsmr-x-{:02d}", i), shape, x.data(), HD5::Dims::Image);
  };
  LSMR lsmr{A, M, nullptr, lsqOpts.its.Get(), lsqOpts.atol.Get(), lsqOpts.btol.Get(), lsqOpts.ctol.Get(), debug};

  auto const x = lsmr.run(CollapseToConstVector(noncart), lsqOpts.λ.Get());
  auto const xm = AsTensorMap(x, A->ishape);

  TOps::Pad<Cx, 5> oc(traj.matrixForFOV(cropFov.Get(), A->ishape[0], nT), A->ishape);
  auto             out = oc.adjoint(xm);
  if (basis) { basis->applyR(out); }
  WriteOutput(cmd, coreOpts.oname.Get(), out, HD5::Dims::Image, info);
  if (coreOpts.residual) { WriteResidual(cmd, coreOpts.oname.Get(), gridOpts, senseOpts, preOpts, traj, xm, A, noncart); }
  Log::Print(cmd, "Finished");
}
