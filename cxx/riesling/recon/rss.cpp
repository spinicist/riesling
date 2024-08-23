#include "types.hpp"

#include "algo/lsmr.hpp"
#include "io/hd5.hpp"
#include "log.hpp"
#include "op/recon.hpp"
#include "inputs.hpp"
#include "outputs.hpp"
#include "precon.hpp"

using namespace rl;

void main_recon_rss(args::Subparser &parser)
{
  CoreOpts   coreOpts(parser);
  GridOpts   gridOpts(parser);
  PreconOpts preOpts(parser);
  LsqOpts    lsqOpts(parser);

  ParseCommand(parser, coreOpts.iname, coreOpts.oname);

  HD5::Reader reader(coreOpts.iname.Get());
  Info const  info = reader.readInfo();
  Trajectory  traj(reader, info.voxel_size);
  auto const basis = LoadBasis(coreOpts.basisFile.Get());
  Cx5         noncart = reader.readTensor<Cx5>();
  traj.checkDims(FirstN<3>(noncart.dimensions()));
  Index const nC = noncart.dimension(0);
  Index const nS = noncart.dimension(3);
  Index const nT = noncart.dimension(4);

  auto const A = Recon::Channels(coreOpts.ndft, gridOpts, traj, nC, nS, nT, basis.get(), traj.matrixForFOV(coreOpts.fov.Get()));
  auto const M = MakeKspacePre(traj, nC, nT, basis.get(), preOpts.type.Get(), preOpts.bias.Get());
  LSMR const lsmr{A, M, lsqOpts.its.Get(), lsqOpts.atol.Get(), lsqOpts.btol.Get(), lsqOpts.ctol.Get()};
  auto x = lsmr.run(CollapseToConstVector(noncart), lsqOpts.λ.Get());
  auto xm = Tensorfy(x, A->ishape);

  Cx5 const rss = DimDot<1>(xm, xm).sqrt();
  TOps::Crop<Cx, 5> oc(rss.dimensions(), traj.matrixForFOV(coreOpts.fov.Get(), A->ishape[0], nT));
  auto              out = oc.forward(rss);

  WriteOutput(coreOpts.oname.Get(), out, HD5::Dims::Image, info, Log::Saved());
  Log::Print("Finished {}", parser.GetCommand().Name());
}
