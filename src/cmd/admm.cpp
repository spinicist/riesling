#include "types.hpp"

#include "algo/admm-augmented.hpp"
#include "algo/admm.hpp"
#include "algo/lsmr.hpp"
#include "algo/lsqr.hpp"
#include "cropper.h"
#include "func/dict.hpp"
#include "func/llr.hpp"
#include "func/thresh-wavelets.hpp"
#include "io/hd5.hpp"
#include "log.hpp"
#include "op/recon.hpp"
#include "parse_args.hpp"
#include "precond.hpp"
#include "sdc.hpp"
#include "sense.hpp"

using namespace rl;

int main_admm(args::Subparser &parser)
{
  CoreOpts coreOpts(parser);
  SDC::Opts sdcOpts(parser);
  SENSE::Opts senseOpts(parser);

  args::Flag use_cg(parser, "C", "Use CG instead of LSQR for inner loop", {"cg"});
  args::Flag use_lsmr(parser, "L", "Use LSMR instead of LSQR for inner loop", {"lsmr"});

  args::ValueFlag<std::string> pre(parser, "P", "Pre-conditioner (none/kspace/filename)", {"pre"}, "kspace");
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

  args::ValueFlag<Index> wavelets(parser, "W", "Wavelet denoising levels", {"wavelets"}, 4);
  args::ValueFlag<Index> width(parser, "W", "Wavelet width (4/6/8)", {"width", 'w'}, 6);

  ParseCommand(parser, coreOpts.iname);

  HD5::Reader reader(coreOpts.iname.Get());
  Trajectory traj(reader);
  Info const &info = traj.info();
  auto recon = make_recon(coreOpts, sdcOpts, senseOpts, traj, false, reader);
  auto const sz = recon->inputDimensions();

  std::shared_ptr<Prox<Cx4>> reg;
  if (wavelets) {
    reg = std::make_shared<ThresholdWavelets>(sz, width.Get(), wavelets.Get());
  } else {
    reg = std::make_shared<LLR>(patchSize.Get(), true);
  };
  
  Cropper out_cropper(info.matrix, LastN<3>(sz), info.voxel_size, coreOpts.fov.Get());
  Cx4 vol(sz);
  Sz3 outSz = out_cropper.size();
  Cx4 cropped(sz[0], outSz[0], outSz[1], outSz[2]);
  Cx5 allData = reader.readTensor<Cx5>(HD5::Keys::Noncartesian);
  Index const volumes = allData.dimension(4);
  Cx5 out(sz[0], outSz[0], outSz[1], outSz[2], volumes);
  auto const &all_start = Log::Now();

  if (use_cg) {
    auto augmented = make_augmented(recon, ρ.Get());
    ConjugateGradients<AugmentedOp<ReconOp>> cg{augmented, inner_its.Get(), atol.Get()};
    AugmentedADMM<ConjugateGradients<AugmentedOp<ReconOp>>> admm{
      cg, reg.get(), outer_its.Get(), ρ.Get(), abstol.Get(), reltol.Get()};
    for (Index iv = 0; iv < volumes; iv++) {
      out.chip<4>(iv) = out_cropper.crop4(admm.run(recon->adjoint(CChipMap(allData, iv))));
    }
  } else if (use_lsmr) {
    auto M = make_pre(pre.Get(), traj);
    LSMR<ReconOp> lsmr{recon, M, inner_its.Get(), atol.Get(), btol.Get(), ctol.Get(), false};
    ADMM<LSMR<ReconOp>> admm{lsmr, reg, outer_its.Get(), λ.Get(), ρ.Get(), α.Get(), abstol.Get(), reltol.Get()};
    for (Index iv = 0; iv < volumes; iv++) {
      out.chip<4>(iv) = out_cropper.crop4(admm.run(CChipMap(allData, iv)));
    }
  } else {
    auto M = make_pre(pre.Get(), traj);
    LSQR<ReconOp> lsqr{recon, M, inner_its.Get(), atol.Get(), btol.Get(), ctol.Get(), false};
    ADMM<LSQR<ReconOp>> admm{lsqr, reg, outer_its.Get(), λ.Get(), ρ.Get(), α.Get(), abstol.Get(), reltol.Get()};
    for (Index iv = 0; iv < volumes; iv++) {
      out.chip<4>(iv) = out_cropper.crop4(admm.run(CChipMap(allData, iv)));
    }
  }
  Log::Print(FMT_STRING("All Volumes: {}"), Log::ToNow(all_start));
  WriteOutput(out, coreOpts.iname.Get(), coreOpts.oname.Get(), parser.GetCommand().Name(), coreOpts.keepTrajectory, traj);
  return EXIT_SUCCESS;
}
