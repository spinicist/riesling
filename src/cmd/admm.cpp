#include "types.hpp"

#include "algo/admm.hpp"
#include "algo/lsmr.hpp"
#include "cropper.h"
#include "io/hd5.hpp"
#include "log.hpp"
#include "op/recon.hpp"
#include "parse_args.hpp"
#include "precond.hpp"
#include "regularizers.hpp"
#include "scaling.hpp"
#include "sdc.hpp"
#include "sense/sense.hpp"

using namespace rl;

int main_admm(args::Subparser &parser)
{
  CoreOpts    coreOpts(parser);
  GridOpts    GridOpts(parser);
  SDC::Opts   sdcOpts(parser, "none");
  SENSE::Opts senseOpts(parser);
  RegOpts     regOpts(parser);

  args::ValueFlag<std::string> pre(parser, "P", "Pre-conditioner (none/kspace/filename)", {"pre"}, "kspace");
  args::ValueFlag<float>       preBias(parser, "BIAS", "Pre-conditioner Bias (1)", {"pre-bias", 'b'}, 1.f);
  args::ValueFlag<Index>       inner_its(parser, "ITS", "Max inner iterations (1)", {"max-its"}, 1);
  args::ValueFlag<float>       atol(parser, "A", "Tolerance on A", {"atol"}, 1.e-6f);
  args::ValueFlag<float>       btol(parser, "B", "Tolerance on b", {"btol"}, 1.e-6f);
  args::ValueFlag<float>       ctol(parser, "C", "Tolerance on cond(A)", {"ctol"}, 1.e-6f);

  args::ValueFlag<Index> outer_its(parser, "ITS", "ADMM max iterations (20)", {"max-outer-its"}, 20);
  args::ValueFlag<float> ρ(parser, "ρ", "ADMM starting penalty parameter ρ (default 1)", {"rho"}, 1.f);
  args::ValueFlag<float> ε(parser, "ε", "ADMM convergence tolerance (1e-2)", {"eps"}, 1.e-2f);
  args::ValueFlag<float> μ(parser, "μ", "ADMM residual rescaling tolerance (default 1.2)", {"mu"}, 1.2f);
  args::ValueFlag<float> τ(parser, "τ", "ADMM residual rescaling maximum (default 10)", {"tau"}, 10.f);

  ParseCommand(parser, coreOpts.iname);

  HD5::Reader reader(coreOpts.iname.Get());
  Trajectory  traj(reader.readInfo(), reader.readTensor<Re3>(HD5::Keys::Trajectory));
  auto        noncart = reader.readTensor<Cx5>(HD5::Keys::Noncartesian);
  traj.checkDims(FirstN<3>(noncart.dimensions()));
  Index const nV = noncart.dimension(4);

  auto const basis = ReadBasis(coreOpts.basisFile.Get());
  auto const sense = std::make_shared<SenseOp>(SENSE::Choose(senseOpts, GridOpts, traj, noncart), basis.dimension(0));
  auto const recon = make_recon(coreOpts, GridOpts, sdcOpts, traj, sense, basis);
  auto const shape = recon->ishape;
  auto       M = make_kspace_pre(pre.Get(), recon->oshape[0], traj, basis, preBias.Get());

  Cropper    out_cropper(LastN<3>(shape), traj.matrixForFOV(coreOpts.fov.Get()));
  Sz3         outSz = out_cropper.size();
  float const scale = Scaling(coreOpts.scaling, recon, M, &noncart(0, 0, 0, 0, 0));
  noncart.device(Threads::GlobalDevice()) = noncart * noncart.constant(scale);

  Cx5 out(shape[0], outSz[0], outSz[1], outSz[2], nV), resid;
  if (coreOpts.residImage) { resid.resize(shape[0], outSz[0], outSz[1], outSz[2], nV); }

  std::shared_ptr<Ops::Op<Cx>> A = recon; // TGV needs a special A
  Regularizers                 reg(regOpts, shape, A);

  ADMM::DebugX debug_x = [shape](Index const ii, ADMM::Vector const &x) {
    Log::Tensor(fmt::format("admm-x-{:02d}", ii), shape, x.data());
  };

  ADMM::DebugZ debug_z = [&](Index const ii, Index const ir, ADMM::Vector const &Fx, ADMM::Vector const &z,
                             ADMM::Vector const &u) {
    if (std::holds_alternative<Sz4>(reg.sizes[ir])) {
      auto const Fshape = std::get<Sz4>(reg.sizes[ir]);
      Log::Tensor(fmt::format("admm-Fx-{:02d}-{:02d}", ir, ii), Fshape, Fx.data());
      Log::Tensor(fmt::format("admm-z-{:02d}-{:02d}", ir, ii), Fshape, z.data());
      Log::Tensor(fmt::format("admm-u-{:02d}-{:02d}", ir, ii), Fshape, u.data());
    } else if (std::holds_alternative<Sz5>(reg.sizes[ir])) {
      auto const Fshape = std::get<Sz5>(reg.sizes[ir]);
      Log::Tensor(fmt::format("admm-Fx-{:02d}-{:02d}", ir, ii), Fshape, Fx.data());
      Log::Tensor(fmt::format("admm-z-{:02d}-{:02d}", ir, ii), Fshape, z.data());
      Log::Tensor(fmt::format("admm-u-{:02d}-{:02d}", ir, ii), Fshape, u.data());
    }
  };

  ADMM opt{A,       M,       reg.ops, reg.prox, inner_its.Get(), atol.Get(), btol.Get(), ctol.Get(), outer_its.Get(),
           ε.Get(), μ.Get(), τ.Get(), debug_x,  debug_z};

  auto const &all_start = Log::Now();
  for (Index iv = 0; iv < nV; iv++) {
    auto x = reg.ext_x->forward(opt.run(&noncart(0, 0, 0, 0, iv), ρ.Get()));
    auto xm = Tensorfy(x, shape);
    out.chip<4>(iv) = out_cropper.crop4(xm) / out.chip<4>(iv).constant(scale);
    if (coreOpts.residImage || coreOpts.residKSpace) { noncart.chip<4>(iv) -= recon->forward(xm); }
    if (coreOpts.residImage) {
      xm = recon->adjoint(noncart.chip<4>(iv));
      resid.chip<4>(iv) = out_cropper.crop4(xm) / resid.chip<4>(iv).constant(scale);
    }
  }
  Log::Print("All Volumes: {}", Log::ToNow(all_start));
  WriteOutput(coreOpts, out, parser.GetCommand().Name(), traj, Log::Saved(), resid, noncart);
  Log::Print("Finished {}", parser.GetCommand().Name());
  return EXIT_SUCCESS;
}
