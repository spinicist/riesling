#include "algo/lsmr.hpp"
#include "inputs.hpp"
#include "io/hd5.hpp"
#include "log.hpp"
#include "op/recon.hpp"
#include "outputs.hpp"
#include "precon.hpp"

using namespace rl;

void main_channels(args::Subparser &parser)
{
  CoreOpts   coreOpts(parser);
  GridOpts   gridOpts(parser);
  PreconOpts preOpts(parser);
  LsqOpts    lsqOpts(parser);

  ParseCommand(parser, coreOpts.iname, coreOpts.oname);

  HD5::Reader reader(coreOpts.iname.Get());
  Info const  info = reader.readInfo();
  Trajectory  traj(reader, info.voxel_size);
  Basis const basis(coreOpts.basisFile.Get());
  Cx5         noncart = reader.readTensor<Cx5>();
  traj.checkDims(FirstN<3>(noncart.dimensions()));
  Index const nC = noncart.dimension(0);
  Index const nS = noncart.dimension(3);
  Index const nT = noncart.dimension(4);

  auto const A = Recon::Channels(coreOpts.ndft, gridOpts, traj, nC, nS, nT, &basis);
  auto const M = MakeKspacePre(traj, nC, nT, &basis, preOpts.type.Get(), preOpts.bias.Get());
  auto       debug = [&A](Index const i, LSMR::Vector const &x) {
    Log::Tensor(fmt::format("lsmr-x-{:02d}", i), A->ishape, x.data(), {"channel", "v", "x", "y", "z"});
  };
  LSMR const lsmr{A, M, lsqOpts.its.Get(), lsqOpts.atol.Get(), lsqOpts.btol.Get(), lsqOpts.ctol.Get(), debug};

  auto const x = lsmr.run(noncart.data(), lsqOpts.λ.Get());
  auto const xm = Tensorfy(x, A->ishape);

  TOps::Crop<Cx, 6> oc(A->ishape, AddFront(traj.matrixForFOV(coreOpts.fov.Get(), A->ishape[1], nT), nC));
  auto              out = oc.forward(xm);
  WriteOutput(coreOpts.oname.Get(), out, HD5::Dims::Channels, info, Log::Saved());
  if (coreOpts.residual) { WriteResidual(coreOpts.residual.Get(), noncart, xm, info, A, M, HD5::Dims::Channels); }
  Log::Print("Finished {}", parser.GetCommand().Name());
}
