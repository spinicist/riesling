#include "types.hpp"

#include "algo/admm.hpp"
#include "algo/lsmr.hpp"
#include "inputs.hpp"
#include "io/hd5.hpp"
#include "log.hpp"
#include "op/recon.hpp"
#include "outputs.hpp"
#include "precon.hpp"
#include "regularizers.hpp"
#include "scaling.hpp"
#include "sense/sense.hpp"

using namespace rl;

void main_recon_rlsq(args::Subparser &parser)
{
  CoreOpts               coreOpts(parser);
  GridOpts               gridOpts(parser);
  PreconOpts             preOpts(parser);
  SENSE::Opts            senseOpts(parser);
  RlsqOpts               rlsqOpts(parser);
  RegOpts                regOpts(parser);
  args::ValueFlag<Index> debugIters(parser, "I", "Write debug images ever N outer iterations (10)", {"debug-iters"}, 10);
  args::Flag             debugZ(parser, "Z", "Write regularizer debug images", {"debug-z"});

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
  auto const recon = Recon::Choose(gridOpts, senseOpts, traj, basis.get(), noncart);
  auto const shape = recon->ishape;
  auto const M = MakeKspacePre(traj, nC, nS, nT, basis.get(), preOpts.type.Get(), preOpts.bias.Get());

  auto [reg, A, ext_x] = Regularizers(regOpts, recon);

  ADMM::DebugX debug_x = [shape, di = debugIters.Get()](Index const ii, ADMM::Vector const &x) {
    if (ii % di == 0) { Log::Tensor(fmt::format("admm-x-{:02d}", ii), shape, x.data(), HD5::Dims::Image); }
  };

  ADMM::DebugZ debug_z = [&, di = debugIters.Get()](Index const ii, Index const ir, ADMM::Vector const &Fx,
                                                    ADMM::Vector const &z, ADMM::Vector const &u) {
    if (debugZ && (ii % di == 0)) {
      if (std::holds_alternative<Sz5>(reg[ir].size)) {
        auto const Fshape = std::get<Sz5>(reg[ir].size);
        Log::Tensor(fmt::format("admm-Fx-{:02d}-{:02d}", ir, ii), Fshape, Fx.data(), HD5::Dims::Image);
        Log::Tensor(fmt::format("admm-z-{:02d}-{:02d}", ir, ii), Fshape, z.data(), HD5::Dims::Image);
        Log::Tensor(fmt::format("admm-u-{:02d}-{:02d}", ir, ii), Fshape, u.data(), HD5::Dims::Image);
      }
    }
  };

  ADMM opt{A,
           M,
           reg,
           rlsqOpts.inner_its0.Get(),
           rlsqOpts.inner_its1.Get(),
           rlsqOpts.atol.Get(),
           rlsqOpts.btol.Get(),
           rlsqOpts.ctol.Get(),
           rlsqOpts.outer_its.Get(),
           rlsqOpts.ε.Get(),
           rlsqOpts.μ.Get(),
           rlsqOpts.τ.Get(),
           debug_x,
           debug_z};

  auto const x = ext_x->forward(opt.run(CollapseToConstVector(noncart), rlsqOpts.ρ.Get()));
  auto const xm = Tensorfy(x, recon->ishape);

  TOps::Crop<Cx, 5> oc(recon->ishape, traj.matrixForFOV(gridOpts.fov.Get(), recon->ishape[0], nT));
  auto              out = oc.forward(xm);
  if (basis) { basis->applyR(out); }
  WriteOutput(cmd, coreOpts.oname.Get(), out, HD5::Dims::Image, info);
  if (coreOpts.residual) { WriteResidual(cmd, coreOpts.residual.Get(), noncart, xm, info, recon, M, HD5::Dims::Image); }
  Log::Print(cmd, "Finished");
}
