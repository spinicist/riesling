#include "types.h"

#include "algo/admm-augmented.hpp"
#include "algo/admm.hpp"
#include "algo/llr.h"
#include "algo/lsmr.hpp"
#include "algo/lsqr.hpp"
#include "cropper.h"
#include "io/hd5.hpp"
#include "log.h"
#include "op/recon.hpp"
#include "parse_args.h"
#include "precond/single.hpp"
#include "sdc.h"
#include "sense.h"

using namespace rl;

int main_admm(args::Subparser &parser)
{
  CoreOpts core(parser);
  ExtraOpts extra(parser);
  SDC::Opts sdcOpts(parser);
  SENSE::Opts senseOpts(parser);

  args::Flag use_cg(parser, "C", "Use CG instead of LSQR for inner loop", {"cg"});
  args::Flag use_lsmr(parser, "L", "Use LSMR instead of LSQR for inner loop", {"lsmr"});

  args::Flag precond(parser, "P", "Apply Ong's single-channel M-conditioner", {"pre"});
  args::ValueFlag<Index> inner_its(parser, "ITS", "Max inner iterations (2)", {"max-its"}, 8);
  args::ValueFlag<float> atol(parser, "A", "Tolerance on A", {"atol"}, 1.e-6f);
  args::ValueFlag<float> btol(parser, "B", "Tolerance on b", {"btol"}, 1.e-6f);
  args::ValueFlag<float> ctol(parser, "C", "Tolerance on cond(A)", {"ctol"}, 1.e-6f);

  args::ValueFlag<Index> outer_its(parser, "ITS", "ADMM max iterations (8)", {"max-outer-its"}, 8);
  args::ValueFlag<float> abstol(parser, "ABS", "ADMM absolute tolerance (1e-3)", {"abs-tol"}, 1.e-3f);
  args::ValueFlag<float> reltol(parser, "REL", "ADMM relative tolerance (1e-3)", {"rel-tol"}, 1.e-3f);
  args::ValueFlag<float> ρ(parser, "ρ", "ADMM Langrangian ρ (default 0.1)", {"rho"}, 0.1f);
  args::ValueFlag<float> α(parser, "α", "ADMM relaxation α (default 1)", {"relax"}, 1.f);

  args::ValueFlag<float> λ(parser, "λ", "Regularization parameter (default 0.1)", {"lambda"}, 0.1f);
  args::ValueFlag<Index> patchSize(parser, "SZ", "Patch size (default 4)", {"patch-size"}, 4);

  ParseCommand(parser, core.iname);

  HD5::RieslingReader reader(core.iname.Get());
  Trajectory const traj = reader.trajectory();
  Info const &info = traj.info();
  auto const kernel = rl::make_kernel(core.ktype.Get(), info.type, core.osamp.Get());
  Mapping const mapping(reader.trajectory(), kernel.get(), core.osamp.Get(), core.bucketSize.Get());
  auto const basis = ReadBasis(core.basisFile);
  auto gridder = make_grid<Cx>(kernel.get(), mapping, info.channels, basis);
  auto const sdc = SDC::Choose(sdcOpts, traj, core.osamp.Get());
  Cx4 senseMaps = SENSE::Choose(senseOpts, info, gridder.get(), extra.iter_fov.Get(), sdc.get(), reader);
  ReconOp recon(gridder.get(), senseMaps, sdc.get());
  std::unique_ptr<Precond<Cx3>> M = precond ? std::make_unique<SingleChannel>(traj, kernel.get(), basis) : nullptr;
  auto reg = [&](Cx4 const &x) -> Cx4 { return llr_sliding(x, λ.Get() / ρ.Get(), patchSize.Get()); };

  auto sz = recon.inputDimensions();
  Cropper out_cropper(info, LastN<3>(sz), extra.out_fov.Get());
  Cx4 vol(sz);
  Sz3 outSz = out_cropper.size();
  Cx4 cropped(sz[0], outSz[0], outSz[1], outSz[2]);
  Cx5 out(sz[0], outSz[0], outSz[1], outSz[2], info.volumes);
  auto const &all_start = Log::Now();

  if (use_cg) {
    AugmentedOp<ReconOp> augmented{recon, ρ.Get()};
    ConjugateGradients<AugmentedOp<ReconOp>> cg{augmented, inner_its.Get(), atol.Get()};
    AugmentedADMM<ConjugateGradients<AugmentedOp<ReconOp>>> admm{
      cg, reg, outer_its.Get(), ρ.Get(), abstol.Get(), reltol.Get()};
    for (Index iv = 0; iv < info.volumes; iv++) {
      out.chip<4>(iv) = out_cropper.crop4(admm.run(recon.Adj(reader.noncartesian(iv))));
    }
  } else if (use_lsmr) {
    LSMR<ReconOp> lsmr{recon, M.get(), inner_its.Get(), atol.Get(), btol.Get(), ctol.Get(), ρ.Get(), false};
    ADMM<LSMR<ReconOp>> admm{lsmr, reg, outer_its.Get(), ρ.Get(), α.Get(), abstol.Get(), reltol.Get()};
    for (Index iv = 0; iv < info.volumes; iv++) {
      out.chip<4>(iv) = out_cropper.crop4(admm.run(reader.noncartesian(iv)));
    }
  } else {
    LSQR<ReconOp> lsqr{recon, M.get(), nullptr, inner_its.Get(), atol.Get(), btol.Get(), ctol.Get(), ρ.Get(), false};
    ADMM<LSQR<ReconOp>> admm{lsqr, reg, outer_its.Get(), ρ.Get(), α.Get(), abstol.Get(), reltol.Get()};
    for (Index iv = 0; iv < info.volumes; iv++) {
      out.chip<4>(iv) = out_cropper.crop4(admm.run(reader.noncartesian(iv)));
    }
  }
  Log::Print(FMT_STRING("All Volumes: {}"), Log::ToNow(all_start));
  WriteOutput(out, core.iname.Get(), core.oname.Get(), parser.GetCommand().Name(), core.keepTrajectory, traj);
  return EXIT_SUCCESS;
}
