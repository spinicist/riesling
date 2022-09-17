#include "types.hpp"

#include "algo/admm-augmented.hpp"
#include "algo/admm.hpp"
#include "algo/lsmr.hpp"
#include "algo/lsqr.hpp"
#include "cropper.h"
#include "func/dict.hpp"
#include "func/llr.hpp"
#include "func/pre-kspace.hpp"
#include "io/hd5.hpp"
#include "log.hpp"
#include "op/recon.hpp"
#include "parse_args.hpp"
#include "sdc.hpp"
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
  args::ValueFlag<Index> patchSize(parser, "SZ", "Patch size for LLR (default 4)", {"patch-size"}, 4);
  args::Flag dictReg(parser, "D", "Use dictionary projection as regularizer", {"dict"});

  ParseCommand(parser, core.iname);

  HD5::RieslingReader reader(core.iname.Get());
  auto const traj = reader.trajectory();
  Info const &info = traj.info();
  auto const basis = ReadBasis(core.basisFile);
  Index const channels = reader.dimensions<4>(HD5::Keys::Noncartesian)[0];
  Index const volumes = reader.dimensions<4>(HD5::Keys::Noncartesian)[3];
  auto gridder = make_grid<Cx, 3>(traj, core.ktype.Get(), core.osamp.Get(), channels, basis);
  auto const sdc = SDC::Choose(sdcOpts, traj, channels, core.ktype.Get(), core.osamp.Get());
  Cx4 senseMaps = SENSE::Choose(senseOpts, traj, gridder.get(), extra.iter_fov.Get(), sdc.get(), reader);
  ReconOp recon(gridder.get(), senseMaps, sdc.get());
  std::unique_ptr<Functor<Cx3>> M = precond ? std::make_unique<KSpaceSingle>(traj) : nullptr;
  std::unique_ptr<Functor<Cx4>> reg;
  if (dictReg) {
    HD5::Reader dict(core.basisFile.Get());
    reg = std::make_unique<DictionaryProjection>(dict.readTensor<Re2>(HD5::Keys::Dictionary));
  } else {
    reg = std::make_unique<LLR>(λ.Get(), patchSize.Get(), true);
  };
  auto sz = recon.inputDimensions();
  Cropper out_cropper(info.matrix, LastN<3>(sz), info.voxel_size, extra.out_fov.Get());
  Cx4 vol(sz);
  Sz3 outSz = out_cropper.size();
  Cx4 cropped(sz[0], outSz[0], outSz[1], outSz[2]);
  Cx5 out(sz[0], outSz[0], outSz[1], outSz[2], volumes);
  auto const &all_start = Log::Now();

  if (use_cg) {
    AugmentedOp<ReconOp> augmented{recon, ρ.Get()};
    ConjugateGradients<AugmentedOp<ReconOp>> cg{augmented, inner_its.Get(), atol.Get()};
    AugmentedADMM<ConjugateGradients<AugmentedOp<ReconOp>>> admm{
      cg, reg.get(), outer_its.Get(), ρ.Get(), abstol.Get(), reltol.Get()};
    for (Index iv = 0; iv < volumes; iv++) {
      out.chip<4>(iv) = out_cropper.crop4(admm.run(recon.adjoint(reader.noncartesian(iv))));
    }
  } else if (use_lsmr) {
    LSMR<ReconOp> lsmr{recon, M.get(), inner_its.Get(), atol.Get(), btol.Get(), ctol.Get(), ρ.Get(), false};
    ADMM<LSMR<ReconOp>> admm{lsmr, reg.get(), outer_its.Get(), ρ.Get(), α.Get(), abstol.Get(), reltol.Get()};
    for (Index iv = 0; iv < volumes; iv++) {
      out.chip<4>(iv) = out_cropper.crop4(admm.run(reader.noncartesian(iv)));
    }
  } else {
    LSQR<ReconOp> lsqr{recon, M.get(), inner_its.Get(), atol.Get(), btol.Get(), ctol.Get(), ρ.Get(), false};
    ADMM<LSQR<ReconOp>> admm{lsqr, reg.get(), outer_its.Get(), ρ.Get(), α.Get(), abstol.Get(), reltol.Get()};
    for (Index iv = 0; iv < volumes; iv++) {
      out.chip<4>(iv) = out_cropper.crop4(admm.run(reader.noncartesian(iv)));
    }
  }
  Log::Print(FMT_STRING("All Volumes: {}"), Log::ToNow(all_start));
  WriteOutput(out, core.iname.Get(), core.oname.Get(), parser.GetCommand().Name(), core.keepTrajectory, traj);
  return EXIT_SUCCESS;
}
