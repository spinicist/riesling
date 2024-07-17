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
  CoreOpts    coreOpts(parser);
  GridOpts    gridOpts(parser);
  PreconOpts  preOpts(parser);
  SENSE::Opts senseOpts(parser);
  RlsqOpts    rlsqOpts(parser);
  RegOpts     regOpts(parser);

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
  auto const recon = Recon::SENSE(coreOpts.ndft, gridOpts, senseOpts, traj, nS, nT, basis, noncart);
  auto const shape = recon->ishape;
  auto const M = MakeKspacePre(traj, nC, nT, basis, preOpts.type.Get(), preOpts.bias.Get());

  auto [reg, A, ext_x] = Regularizers(regOpts, recon);

  ADMM::DebugX debug_x = [shape](Index const ii, ADMM::Vector const &x) {
    Log::Tensor(fmt::format("admm-x-{:02d}", ii), shape, x.data());
  };

  ADMM::DebugZ debug_z = [&](Index const ii, Index const ir, ADMM::Vector const &Fx, ADMM::Vector const &z,
                             ADMM::Vector const &u) {
    if (std::holds_alternative<Sz4>(reg[ir].size)) {
      auto const Fshape = std::get<Sz4>(reg[ir].size);
      Log::Tensor(fmt::format("admm-Fx-{:02d}-{:02d}", ir, ii), Fshape, Fx.data());
      Log::Tensor(fmt::format("admm-z-{:02d}-{:02d}", ir, ii), Fshape, z.data());
      Log::Tensor(fmt::format("admm-u-{:02d}-{:02d}", ir, ii), Fshape, u.data());
    } else if (std::holds_alternative<Sz5>(reg[ir].size)) {
      auto const Fshape = std::get<Sz5>(reg[ir].size);
      Log::Tensor(fmt::format("admm-Fx-{:02d}-{:02d}", ir, ii), Fshape, Fx.data());
      Log::Tensor(fmt::format("admm-z-{:02d}-{:02d}", ir, ii), Fshape, z.data());
      Log::Tensor(fmt::format("admm-u-{:02d}-{:02d}", ir, ii), Fshape, u.data());
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

  auto const x = ext_x->forward(opt.run(noncart.data(), rlsqOpts.ρ.Get()));
  auto const xm = Tensorfy(x, recon->ishape);

  TOps::Crop<Cx, 5> oc(recon->ishape, traj.matrixForFOV(coreOpts.fov.Get(), recon->ishape[0], nT));
  auto              out = oc.forward(xm);
  WriteOutput(coreOpts.oname.Get(), out, HD5::Dims::Image, info, Log::Saved());
  if (coreOpts.residual) { WriteResidual(coreOpts.residual.Get(), noncart, xm, info, recon, M, HD5::Dims::Image); }
  Log::Print("Finished {}", parser.GetCommand().Name());
}
