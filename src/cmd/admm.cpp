#include "types.hpp"

#include "algo/admm.hpp"
#include "algo/lsmr.hpp"
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
  args::Flag tv(parser, "TV", "Total Variation", {"tv"});
  args::Flag tgv(parser, "TGV", "Total Generalized Variation", {"tgv"});
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
  auto M = make_pre(pre.Get(), recon->oshape, traj, ReadBasis(coreOpts.basisFile.Get()), preBias.Get());
  auto const sz = recon->ishape;

  Cropper out_cropper(info.matrix, LastN<3>(sz), info.voxel_size, coreOpts.fov.Get());
  Sz3 outSz = out_cropper.size();
  Cx5 allData = reader.readTensor<Cx5>(HD5::Keys::Noncartesian);
  float const scale = Scaling(coreOpts.scaling, recon, M->adjoint(CChipMap(allData, 0)));
  allData.device(Threads::GlobalDevice()) = allData * allData.constant(scale);
  Index const volumes = allData.dimension(4);
  Cx5 out(sz[0], outSz[0], outSz[1], outSz[2], volumes);

  std::vector<std::shared_ptr<LinOps::Op<Cx>>> reg_ops;
  std::vector<std::shared_ptr<Prox<Cx>>> prox;
  std::shared_ptr<LinOps::Op<Cx>> A, ext_x; // Need these for TGV, sigh
  A = recon;
  if (wavelets) {
    prox.push_back(std::make_shared<ThresholdWavelets>(λ.Get(), sz, width.Get(), wavelets.Get()));
    reg_ops.push_back(std::make_shared<TensorIdentity<Cx, 4>>(sz));
  } else if (patchSize) {
    prox.push_back(std::make_shared<LLR>(λ.Get(), patchSize.Get(), winSize.Get(), sz));
    reg_ops.push_back(std::make_shared<TensorIdentity<Cx, 4>>(sz));
  } else if (nmrent) {
    prox.push_back(std::make_shared<NMREntropy>(λ.Get()));
    reg_ops.push_back(std::make_shared<TensorIdentity<Cx, 4>>(sz));
  } else if (l1) {
    prox.push_back(std::make_shared<SoftThreshold>(λ.Get()));
    reg_ops.push_back(std::make_shared<TensorIdentity<Cx, 4>>(sz));
  } else if (tv) {
    prox.push_back(std::make_shared<SoftThreshold>(λ.Get()));
    reg_ops.push_back(std::make_shared<GradOp>(sz));
  } else if (tgv) {
    auto grad_x = std::make_shared<GradOp>(sz);
    ext_x = std::make_shared<LinOps::Extract<Cx>>(recon->cols() + grad_x->rows(), 0, recon->cols());
    auto ext_v = std::make_shared<LinOps::Extract<Cx>>(recon->cols() + grad_x->rows(), recon->cols(), grad_x->rows());
    auto op1 = std::make_shared<LinOps::Subtract<Cx>>(std::make_shared<LinOps::Multiply<Cx>>(grad_x, ext_x), ext_v);
    auto prox1 = std::make_shared<SoftThreshold>(λ.Get());
    auto grad_v = std::make_shared<GradVecOp>(grad_x->oshape);
    auto op2 = std::make_shared<LinOps::Multiply<Cx>>(grad_v, ext_v);
    auto prox2 = std::make_shared<SoftThreshold>(λ.Get());
    prox = {prox1, prox2};
    reg_ops = {op1, op2};
    A = std::make_shared<LinOps::Multiply<Cx>>(recon, ext_x);
  } else {
    Log::Fail("Must specify at least one regularizer");
  }

  ADMM admm{
    A,
    M,
    inner_its.Get(),
    atol.Get(),
    btol.Get(),
    ctol.Get(),
    reg_ops,
    prox,
    outer_its.Get(),
    α.Get(),
    μ.Get(),
    τ.Get(),
    abstol.Get(),
    reltol.Get()};
  for (Index iv = 0; iv < volumes; iv++) {
    auto x = admm.run(&allData(0, 0, 0, 0, iv), ρ.Get());
    if (ext_x) {
      x = ext_x->forward(x);
    }
    out.chip<4>(iv) = out_cropper.crop4(Tensorfy(x, sz)) / out.chip<4>(iv).constant(scale);
  }

  auto const &all_start = Log::Now();

  Log::Print("All Volumes: {}", Log::ToNow(all_start));
  WriteOutput(coreOpts, out, parser.GetCommand().Name(), traj, Log::Saved());
  return EXIT_SUCCESS;
}
