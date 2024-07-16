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

  ParseCommand(parser, coreOpts.iname, coreOpts.oname);

  HD5::Reader reader(coreOpts.iname.Get());
  Info const  info = reader.readInfo();
  Trajectory  traj(reader, info.voxel_size);
  auto        noncart = reader.readTensor<Cx5>();
  traj.checkDims(FirstN<3>(noncart.dimensions()));
  Index const nC = noncart.dimension(0);
  Index const nS = noncart.dimension(3);
  Index const nT = noncart.dimension(4);

  auto const basis = ReadBasis(coreOpts.basisFile.Get());
  auto const A = Recon::SENSE(coreOpts.ndft, gridOpts, senseOpts, traj, nS, nT, basis, noncart);
  auto const M = MakeKspacePre(traj, nC, nT, basis, preOpts.type.Get(), preOpts.bias.Get(), coreOpts.ndft.Get());
  Log::Print("A {} {} M {} {}", A->ishape, A->oshape, M->rows(), M->cols());
  auto debug = [&A](Index const i, LSMR::Vector const &x) {
    Log::Tensor(fmt::format("lsmr-x-{:02d}", i), A->ishape, x.data(), {"v", "x", "y", "z"});
  };
  LSMR lsmr{A, M, lsqOpts.its.Get(), lsqOpts.atol.Get(), lsqOpts.btol.Get(), lsqOpts.ctol.Get(), debug};
  auto x = lsmr.run(noncart.data(), lsqOpts.λ.Get());
  auto xm = Tensorfy(x, A->ishape);

  TOps::Crop<Cx, 5> oc(A->ishape, traj.matrixForFOV(coreOpts.fov.Get(), A->ishape[0], nT));
  auto              out = oc.forward(xm);
  WriteOutput(coreOpts.oname.Get(), out, info, Log::Saved());
  if (coreOpts.residual) { WriteResidual(coreOpts.residual.Get(), noncart, xm, info, A, M); }
  Log::Print("Finished {}", parser.GetCommand().Name());
}
