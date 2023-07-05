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
  CoreOpts coreOpts(parser);
  SDC::Opts sdcOpts(parser, "none");
  SENSE::Opts senseOpts(parser);
  RegOpts regOpts(parser);

  args::ValueFlag<std::string> pre(parser, "P", "Pre-conditioner (none/kspace/filename)", {"pre"}, "kspace");
  args::ValueFlag<Index> inner_its(parser, "ITS", "Max inner iterations (2)", {"max-its"}, 2);
  args::ValueFlag<float> atol(parser, "A", "Tolerance on A", {"atol"}, 1.e-6f);
  args::ValueFlag<float> btol(parser, "B", "Tolerance on b", {"btol"}, 1.e-6f);
  args::ValueFlag<float> ctol(parser, "C", "Tolerance on cond(A)", {"ctol"}, 1.e-6f);

  args::ValueFlag<Index> outer_its(parser, "ITS", "ADMM max iterations (30)", {"max-outer-its"}, 30);
  args::ValueFlag<float> abstol(parser, "ABS", "ADMM absolute tolerance (1e-4)", {"abs-tol"}, 1.e-4f);
  args::ValueFlag<float> reltol(parser, "REL", "ADMM relative tolerance (1e-3)", {"rel-tol"}, 1.e-3f);
  args::ValueFlag<float> ρ(parser, "ρ", "ADMM penalty parameter ρ (default 1)", {"rho"}, 1.f);
  args::ValueFlag<float> α(parser, "α", "ADMM relaxation α (default 1)", {"relax"}, 1.f);
  args::ValueFlag<float> μ(parser, "μ", "ADMM primal-dual mismatch limit (10)", {"mu"}, 10.f);
  args::ValueFlag<float> τ(parser, "τ", "ADMM primal-dual rescale (2)", {"tau"}, 2.f);
  args::Flag hogwild(parser, "HW", "Use Hogwild scheme", {"hogwild"});

  ParseCommand(parser, coreOpts.iname);

  HD5::Reader reader(coreOpts.iname.Get());
  Trajectory traj(reader.readInfo(), reader.readTensor<Re3>(HD5::Keys::Trajectory));
  Info const &info = traj.info();
  auto noncart = reader.readTensor<Cx5>(HD5::Keys::Noncartesian);
  Index const nV = noncart.dimension(4);

  auto const basis = ReadBasis(coreOpts.basisFile.Get());
  auto const sense = std::make_shared<SenseOp>(SENSE::Choose(senseOpts, coreOpts, traj, noncart), basis.dimension(0));
  auto const recon = make_recon(coreOpts, sdcOpts, traj, sense, basis);
  auto const shape = recon->ishape;
  auto M = make_kspace_pre(pre.Get(), recon->oshape[0], traj, ReadBasis(coreOpts.basisFile.Get()));

  Cropper out_cropper(info.matrix, LastN<3>(shape), info.voxel_size, coreOpts.fov.Get());
  Sz3 outSz = out_cropper.size();
  float const scale = Scaling(coreOpts.scaling, recon, M, &noncart(0, 0, 0, 0, 0));
  noncart.device(Threads::GlobalDevice()) = noncart * noncart.constant(scale);

  Cx5 out(shape[0], outSz[0], outSz[1], outSz[2], nV), resid;
  if (coreOpts.residImage) { resid.resize(shape[0], outSz[0], outSz[1], outSz[2], nV); }

  std::shared_ptr<Ops::Op<Cx>> A = recon; // TGV needs a special A
  Regularizers reg(regOpts, shape, A);

  std::function<void(Index const, ADMM::Vector const &)> debug_x = [shape](Index const ii, ADMM::Vector const &x) {
    Log::Tensor(fmt::format("admm-x-{:02d}", ii), shape, x.data());
  };

  std::function<void(Index const, Index const, ADMM::Vector const &)> debug_z =
    [&](Index const ii, Index const ir, ADMM::Vector const &z) {
      if (auto p = std::dynamic_pointer_cast<TensorOperator<Cx, 4>>(reg.ops[ir])) {
        Log::Tensor(fmt::format("admm-z-{:02d}-{:02d}", ir, ii), p->oshape, z.data());
      }
    };

  ADMM admm{
    A,
    M,
    inner_its.Get(),
    atol.Get(),
    btol.Get(),
    ctol.Get(),
    reg.ops,
    reg.prox,
    outer_its.Get(),
    α.Get(),
    μ.Get(),
    τ.Get(),
    abstol.Get(),
    reltol.Get(),
    hogwild,
    debug_x,
    debug_z};
  auto const &all_start = Log::Now();
  for (Index iv = 0; iv < nV; iv++) {
    auto x = reg.ext_x->forward(admm.run(&noncart(0, 0, 0, 0, iv), ρ.Get()));
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
  return EXIT_SUCCESS;
}
