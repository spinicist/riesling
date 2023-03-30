#include "types.hpp"

#include "algo/admm-augmented.hpp"
#include "algo/admm.hpp"
#include "algo/lsmr.hpp"
#include "algo/lsqr.hpp"
#include "cropper.h"
#include "io/hd5.hpp"
#include "log.hpp"
#include "op/grad.hpp"
#include "op/recon.hpp"
#include "parse_args.hpp"
#include "precond.hpp"
#include "prox/entropy.hpp"
#include "prox/llr.hpp"
#include "prox/thresh-wavelets.hpp"
#include "scaling.hpp"
#include "sdc.hpp"
#include "sense.hpp"

using namespace rl;

int main_admm(args::Subparser &parser)
{
  CoreOpts coreOpts(parser);
  SDC::Opts sdcOpts(parser, "none");
  SENSE::Opts senseOpts(parser);

  args::ValueFlag<std::string> pre(parser, "P", "Pre-conditioner (none/kspace/filename)", {"pre"}, "kspace");
  args::ValueFlag<float> preBias(parser, "BIAS", "Pre-conditioner Bias (1)", {"pre-bias", 'b'}, 1.f);
  args::ValueFlag<Index> inner_its(parser, "ITS", "Max inner iterations (4)", {"max-its"}, 4);
  args::ValueFlag<float> atol(parser, "A", "Tolerance on A", {"atol"}, 1.e-6f);
  args::ValueFlag<float> btol(parser, "B", "Tolerance on b", {"btol"}, 1.e-6f);
  args::ValueFlag<float> ctol(parser, "C", "Tolerance on cond(A)", {"ctol"}, 1.e-6f);

  args::ValueFlag<Index> outer_its(parser, "ITS", "ADMM max iterations (8)", {"max-outer-its"}, 8);
  args::ValueFlag<float> abstol(parser, "ABS", "ADMM absolute tolerance (1e-4)", {"abs-tol"}, 1.e-4f);
  args::ValueFlag<float> reltol(parser, "REL", "ADMM relative tolerance (1e-3)", {"rel-tol"}, 1.e-3f);
  args::ValueFlag<float> ρ(parser, "ρ", "ADMM penalty parameter ρ (default 1)", {"rho"}, 1.f);
  args::ValueFlag<float> α(parser, "α", "ADMM relaxation α (default 1)", {"relax"}, 1.f);
  args::ValueFlag<float> μ(parser, "μ", "ADMM primal-dual mismatch limit (10)", {"mu"}, 10.f);
  args::ValueFlag<float> τ(parser, "τ", "ADMM primal-dual rescale (2)", {"tau"}, 2.f);

  args::ValueFlag<float> λ(parser, "λ", "Regularization parameter (default 1e-3)", {"lambda"}, 1.e-3f);

  // Default is TV on spatial dimensions, i.e. classic compressed sensing
  args::Flag tv4(parser, "TV4", "TV on all 4 dimensions", {"tv4"});
  args::Flag l1(parser, "L1", "Simple L1 regularization", {"l1"});
  args::Flag nmrent(parser, "E", "NMR Entropy", {"nmrent"});

  args::ValueFlag<Index> patchSize(parser, "SZ", "Patch size for LLR (default 4)", {"llr-patch"}, 5);
  args::ValueFlag<Index> winSize(parser, "SZ", "Patch size for LLR (default 4)", {"llr-win"}, 3);

  args::ValueFlag<Index> wavelets(parser, "W", "Wavelet denoising levels", {"wavelets"}, 4);
  args::ValueFlag<Index> width(parser, "W", "Wavelet width (4/6/8)", {"width", 'w'}, 6);

  ParseCommand(parser, coreOpts.iname);

  HD5::Reader reader(coreOpts.iname.Get());
  Trajectory traj(reader);
  Info const &info = traj.info();
  auto recon = make_recon(coreOpts, sdcOpts, senseOpts, traj, false, reader);
  auto M = make_pre(pre.Get(), recon->outputDimensions(), traj, ReadBasis(coreOpts.basisFile.Get()), preBias.Get());
  auto const sz = recon->inputDimensions();

  Cropper out_cropper(info.matrix, LastN<3>(sz), info.voxel_size, coreOpts.fov.Get());
  Sz3 outSz = out_cropper.size();
  Cx5 allData = reader.readTensor<Cx5>(HD5::Keys::Noncartesian);
  float const scale = Scaling(coreOpts.scaling, recon, M->cadjoint(CChipMap(allData, 0)));
  allData.device(Threads::GlobalDevice()) = allData * allData.constant(scale);
  Index const volumes = allData.dimension(4);
  Cx5 out(sz[0], outSz[0], outSz[1], outSz[2], volumes);
  std::map<std::string, float> meta{{"scale", scale}, {"lambda", λ.Get()}, {"rho", ρ.Get()}};

  auto const &all_start = Log::Now();
  if (wavelets) {
    Regularizer<Identity<Cx, 4>> reg{
      .prox = std::make_shared<ThresholdWavelets>(sz, λ.Get(), width.Get(), wavelets.Get()),
      .op = std::make_shared<Identity<Cx, 4>>(sz)};
    LSMR<ReconOp> lsmr{recon, M, inner_its.Get(), atol.Get(), btol.Get(), ctol.Get(), false, reg.op};
    ADMM<LSMR<ReconOp>, Identity<Cx, 4>> admm{
      lsmr, reg, outer_its.Get(), α.Get(), μ.Get(), τ.Get(), abstol.Get(), reltol.Get()};
    for (Index iv = 0; iv < volumes; iv++) {
      out.chip<4>(iv) = out_cropper.crop4(admm.run(CChipMap(allData, iv), ρ.Get())) / out.chip<4>(iv).constant(scale);
    }
  } else if (patchSize) {
    Regularizer<Identity<Cx, 4>> reg{
      .prox = std::make_shared<LLR>(λ.Get(), patchSize.Get(), winSize.Get()), .op = std::make_shared<Identity<Cx, 4>>(sz)};
    LSMR<ReconOp> lsmr{recon, M, inner_its.Get(), atol.Get(), btol.Get(), ctol.Get(), false, reg.op};
    ADMM<LSMR<ReconOp>, Identity<Cx, 4>> admm{
      lsmr, reg, outer_its.Get(), α.Get(), μ.Get(), τ.Get(), abstol.Get(), reltol.Get()};
    for (Index iv = 0; iv < volumes; iv++) {
      out.chip<4>(iv) = out_cropper.crop4(admm.run(CChipMap(allData, iv), ρ.Get())) / out.chip<4>(iv).constant(scale);
    }
  } else if (nmrent) {
    Regularizer<Identity<Cx, 4>> reg{
      .prox = std::make_shared<NMREntropy>(λ.Get()), .op = std::make_shared<Identity<Cx, 4>>(sz)};
    LSMR<ReconOp> lsmr{recon, M, inner_its.Get(), atol.Get(), btol.Get(), ctol.Get(), false, reg.op};
    ADMM<LSMR<ReconOp>, Identity<Cx, 4>> admm{
      lsmr, reg, outer_its.Get(), α.Get(), μ.Get(), τ.Get(), abstol.Get(), reltol.Get()};
    for (Index iv = 0; iv < volumes; iv++) {
      out.chip<4>(iv) = out_cropper.crop4(admm.run(CChipMap(allData, iv), ρ.Get())) / out.chip<4>(iv).constant(scale);
    }
  } else if (l1) {
    Regularizer<Identity<Cx, 4>> reg{
      .prox = std::make_shared<SoftThreshold<Cx4>>(λ.Get()), .op = std::make_shared<Identity<Cx, 4>>(sz)};
    LSMR<ReconOp> lsmr{recon, M, inner_its.Get(), atol.Get(), btol.Get(), ctol.Get(), false, reg.op};
    ADMM<LSMR<ReconOp>, Identity<Cx, 4>> admm{
      lsmr, reg, outer_its.Get(), α.Get(), μ.Get(), τ.Get(), abstol.Get(), reltol.Get()};
    for (Index iv = 0; iv < volumes; iv++) {
      out.chip<4>(iv) = out_cropper.crop4(admm.run(CChipMap(allData, iv), ρ.Get())) / out.chip<4>(iv).constant(scale);
    }
  } else if (tv4) {
    Regularizer<Grad4Op> reg{.prox = std::make_shared<SoftThreshold<Cx5>>(λ.Get()), .op = std::make_shared<Grad4Op>(sz)};
    LSMR<ReconOp, Grad4Op> lsmr{recon, M, inner_its.Get(), atol.Get(), btol.Get(), ctol.Get(), false, reg.op};
    ADMM<LSMR<ReconOp, Grad4Op>, Grad4Op> admm{lsmr, reg, outer_its.Get(), α.Get(), μ.Get(), τ.Get(), abstol.Get(), reltol.Get()};
    for (Index iv = 0; iv < volumes; iv++) {
      out.chip<4>(iv) = out_cropper.crop4(admm.run(CChipMap(allData, iv), ρ.Get())) / out.chip<4>(iv).constant(scale);
    }
  } else {
    Regularizer<GradOp> reg{.prox = std::make_shared<SoftThreshold<Cx5>>(λ.Get()), .op = std::make_shared<GradOp>(sz)};
    LSMR<ReconOp, GradOp> lsmr{recon, M, inner_its.Get(), atol.Get(), btol.Get(), ctol.Get(), false, reg.op};
    ADMM<LSMR<ReconOp, GradOp>, GradOp> admm{lsmr, reg, outer_its.Get(), α.Get(), μ.Get(), τ.Get(), abstol.Get(), reltol.Get()};
    for (Index iv = 0; iv < volumes; iv++) {
      out.chip<4>(iv) = out_cropper.crop4(admm.run(CChipMap(allData, iv), ρ.Get())) / out.chip<4>(iv).constant(scale);
    }
  }

  Log::Print(FMT_STRING("All Volumes: {}"), Log::ToNow(all_start));
  WriteOutput(out, coreOpts.iname.Get(), coreOpts.oname.Get(), parser.GetCommand().Name(), coreOpts.keepTrajectory, traj, meta);
  return EXIT_SUCCESS;
}
