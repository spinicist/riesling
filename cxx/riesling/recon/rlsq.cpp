#include "types.hpp"

#include "algo/admm.hpp"
#include "algo/lsmr.hpp"
#include "io/hd5.hpp"
#include "log.hpp"
#include "op/recon.hpp"
#include "parse_args.hpp"
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
  Index const nS = noncart.dimension(3);
  Index const nV = noncart.dimension(4);

  auto const basis = ReadBasis(coreOpts.basisFile.Get());
  auto const recon = Recon::SENSE(coreOpts, gridOpts, senseOpts, traj, nS, basis, noncart);
  auto const shape = recon->ishape;
  auto const M = make_kspace_pre(traj, recon->oshape[0], basis, gridOpts.vcc, preOpts.type.Get(), preOpts.bias.Get());

  std::shared_ptr<Ops::Op<Cx>> A = recon; // TGV needs a special A
  Regularizers                 reg(regOpts, shape, A);

  ADMM::DebugX debug_x = [shape](Index const ii, ADMM::Vector const &x) {
    Log::Tensor(fmt::format("admm-x-{:02d}", ii), shape, x.data());
  };

  ADMM::DebugZ debug_z = [&](Index const ii, Index const ir, ADMM::Vector const &Fx, ADMM::Vector const &z,
                             ADMM::Vector const &u) {
    if (std::holds_alternative<Sz4>(reg.sizes[static_cast<size_t>(ir)])) {
      auto const Fshape = std::get<Sz4>(reg.sizes[static_cast<size_t>(ir)]);
      Log::Tensor(fmt::format("admm-Fx-{:02d}-{:02d}", ir, ii), Fshape, Fx.data());
      Log::Tensor(fmt::format("admm-z-{:02d}-{:02d}", ir, ii), Fshape, z.data());
      Log::Tensor(fmt::format("admm-u-{:02d}-{:02d}", ir, ii), Fshape, u.data());
    } else if (std::holds_alternative<Sz5>(reg.sizes[static_cast<size_t>(ir)])) {
      auto const Fshape = std::get<Sz5>(reg.sizes[static_cast<size_t>(ir)]);
      Log::Tensor(fmt::format("admm-Fx-{:02d}-{:02d}", ir, ii), Fshape, Fx.data());
      Log::Tensor(fmt::format("admm-z-{:02d}-{:02d}", ir, ii), Fshape, z.data());
      Log::Tensor(fmt::format("admm-u-{:02d}-{:02d}", ir, ii), Fshape, u.data());
    }
  };

  ADMM opt{A,
           M,
           reg.regs,
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

  TOps::Crop<Cx, 4> oc(recon->ishape, AddFront(traj.matrixForFOV(coreOpts.fov.Get()), recon->ishape[0]));
  Cx5               out(AddBack(oc.oshape, nV)), resid;
  if (coreOpts.residual) { resid.resize(out.dimensions()); }

  float const scale = Scaling(rlsqOpts.scaling, recon, M, &noncart(0, 0, 0, 0, 0));
  noncart.device(Threads::GlobalDevice()) = noncart * noncart.constant(scale);

  for (Index iv = 0; iv < nV; iv++) {
    auto x = reg.ext_x->forward(opt.run(&noncart(0, 0, 0, 0, iv), rlsqOpts.ρ.Get()));
    auto xm = Tensorfy(x, recon->ishape);
    out.chip<4>(iv) = oc.forward(xm) / out.chip<4>(iv).constant(scale);
    if (coreOpts.residual) {
      noncart.chip<4>(iv) -= recon->forward(xm);
      xm = recon->adjoint(noncart.chip<4>(iv));
      resid.chip<4>(iv) = oc.forward(xm) / resid.chip<4>(iv).constant(scale);
    }
  }
  WriteOutput(coreOpts.oname.Get(), out, info, Log::Saved());
  if (coreOpts.residual) { WriteOutput(coreOpts.residual.Get(), resid, info); }
  Log::Print("Finished {}", parser.GetCommand().Name());
}
