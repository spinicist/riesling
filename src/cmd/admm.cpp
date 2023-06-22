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
#include "sense/sense.hpp"

using namespace rl;

int main_admm(args::Subparser &parser)
{
  CoreOpts coreOpts(parser);
  SDC::Opts sdcOpts(parser, "none");
  SENSE::Opts senseOpts(parser);

  args::ValueFlag<std::string> pre(parser, "P", "Pre-conditioner (none/kspace/filename)", {"pre"}, "kspace");
  args::ValueFlag<float> preBias(parser, "BIAS", "Pre-conditioner Bias (1)", {"pre-bias", 'b'}, 1.f);
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

  args::ValueFlag<float> tv(parser, "TV", "Total Variation", {"tv"});
  args::ValueFlag<float> tvt(parser, "TVT", "Total Variation along time/frames/basis", {"tvt"});
  args::ValueFlag<float> tgv(parser, "TGV", "Total Generalized Variation", {"tgv"});
  args::ValueFlag<float> l1(parser, "L1", "Simple L1 regularization", {"l1"});
  args::ValueFlag<float> nmrent(parser, "E", "NMR Entropy", {"nmrent"});

  args::ValueFlag<float> llr(parser, "L", "LLR regularization", {"llr"});
  args::ValueFlag<Index> llrPatch(parser, "SZ", "Patch size for LLR (default 4)", {"llr-patch"}, 5);
  args::ValueFlag<Index> llrWin(parser, "SZ", "Patch size for LLR (default 4)", {"llr-win"}, 3);

  args::ValueFlag<Index> wavelets(parser, "L", "L1 Wavelet denoising", {"wavelets"});
  args::ValueFlag<Index> waveLevels(parser, "W", "Wavelet denoising levels", {"wavelet-levels"}, 4);
  args::ValueFlag<Index> waveWidth(parser, "W", "Wavelet width (4/6/8)", {"wavelet-width"}, 6);

  ParseCommand(parser, coreOpts.iname);

  HD5::Reader reader(coreOpts.iname.Get());
  Trajectory traj(reader);
  Info const &info = traj.info();
  auto recon = make_recon(coreOpts, sdcOpts, senseOpts, traj, reader);
  std::shared_ptr<Ops::Op<Cx>> A = recon; // TGV needs a special A
  auto const sz = recon->ishape;
  auto M = make_kspace_pre(pre.Get(), recon->oshape[0], traj, ReadBasis(coreOpts.basisFile.Get()), preBias.Get());

  Cropper out_cropper(info.matrix, LastN<3>(sz), info.voxel_size, coreOpts.fov.Get());
  Sz3 outSz = out_cropper.size();
  Cx5 allData = reader.readTensor<Cx5>(HD5::Keys::Noncartesian);
  float const scale = Scaling(coreOpts.scaling, recon, M, &allData(0, 0, 0, 0, 0));
  allData.device(Threads::GlobalDevice()) = allData * allData.constant(scale);
  Index const volumes = allData.dimension(4);
  Cx5 out(sz[0], outSz[0], outSz[1], outSz[2], volumes), resid;
  if (coreOpts.residImage) {
    resid.resize(sz[0], outSz[0], outSz[1], outSz[2], volumes);
  }

  std::vector<std::shared_ptr<Ops::Op<Cx>>> reg_ops;
  std::vector<std::shared_ptr<Prox<Cx>>> prox;
  std::shared_ptr<Ops::Op<Cx>> ext_x = std::make_shared<TensorIdentity<Cx, 4>>(sz); // Need for TGV, sigh
  std::function<void(Index const, ADMM::Vector const &)> debug_x = [sz](Index const ii, ADMM::Vector const &x) {
    Log::Tensor(fmt::format("admm-x-{:02d}", ii), sz, x.data());
  };

  if (tgv) {
    auto grad_x = std::make_shared<GradOp>(sz, std::vector<Index>{1, 2, 3});
    ext_x = std::make_shared<Ops::Extract<Cx>>(A->cols() + grad_x->rows(), 0, A->cols());
    auto ext_v = std::make_shared<Ops::Extract<Cx>>(A->cols() + grad_x->rows(), A->cols(), grad_x->rows());
    auto op1 = std::make_shared<Ops::Subtract<Cx>>(std::make_shared<Ops::Multiply<Cx>>(grad_x, ext_x), ext_v);
    auto prox1 = std::make_shared<SoftThreshold>(tgv.Get(), op1->rows());
    auto grad_v = std::make_shared<GradVecOp>(grad_x->oshape);
    auto op2 = std::make_shared<Ops::Multiply<Cx>>(grad_v, ext_v);
    auto prox2 = std::make_shared<SoftThreshold>(tgv.Get(), op2->rows());
    prox = {prox1, prox2};
    reg_ops = {op1, op2};
    A = std::make_shared<Ops::Multiply<Cx>>(A, ext_x);
    debug_x = [sz, grad_x](Index const ii, ADMM::Vector const &xv) {
      Log::Tensor(fmt::format("admm-x-{:02d}", ii), sz, xv.data());
      Log::Tensor(fmt::format("admm-v-{:02d}", ii), grad_x->oshape, xv.data() + Product(sz));
    };
  }

  if (wavelets) {
    reg_ops.push_back(std::make_shared<Ops::Multiply<Cx>>(std::make_shared<TensorIdentity<Cx, 4>>(sz), ext_x));
    prox.push_back(std::make_shared<ThresholdWavelets>(wavelets.Get(), sz, waveWidth.Get(), waveLevels.Get()));
  }

  if (llr) {
    reg_ops.push_back(std::make_shared<Ops::Multiply<Cx>>(std::make_shared<TensorIdentity<Cx, 4>>(sz), ext_x));
    prox.push_back(std::make_shared<LLR>(llr.Get(), llrPatch.Get(), llrWin.Get(), sz));
  }

  if (nmrent) {
    reg_ops.push_back(std::make_shared<Ops::Multiply<Cx>>(std::make_shared<TensorIdentity<Cx, 4>>(sz), ext_x));
    prox.push_back(std::make_shared<NMREntropy>(nmrent.Get(), reg_ops.back()->rows()));
  }

  if (l1) {
    reg_ops.push_back(std::make_shared<Ops::Multiply<Cx>>(std::make_shared<TensorIdentity<Cx, 4>>(sz), ext_x));
    prox.push_back(std::make_shared<SoftThreshold>(l1.Get(), reg_ops.back()->rows()));
  }

  if (tv) {
    reg_ops.push_back(std::make_shared<Ops::Multiply<Cx>>(std::make_shared<GradOp>(sz, std::vector<Index>{1, 2, 3}), ext_x));
    prox.push_back(std::make_shared<SoftThreshold>(tv.Get(), reg_ops.back()->rows()));
  }

  if (tvt) {
    reg_ops.push_back(std::make_shared<Ops::Multiply<Cx>>(std::make_shared<GradOp>(sz, std::vector<Index>{0}), ext_x));
    prox.push_back(std::make_shared<SoftThreshold>(tvt.Get(), reg_ops.back()->rows()));
  }

  if (prox.size() == 0) {
    Log::Fail("Must specify at least one regularizer");
  }

  std::function<void(Index const, Index const, ADMM::Vector const &)> debug_z =
    [&](Index const ii, Index const ir, ADMM::Vector const &z) {
      if (auto p = std::dynamic_pointer_cast<TensorOperator<Cx, 4>>(reg_ops[ir])) {
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
    reg_ops,
    prox,
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
  for (Index iv = 0; iv < volumes; iv++) {
    auto x = ext_x->forward(admm.run(&allData(0, 0, 0, 0, iv), ρ.Get()));
    auto xm = Tensorfy(x, sz);
    out.chip<4>(iv) = out_cropper.crop4(xm) / out.chip<4>(iv).constant(scale);
    if (coreOpts.residImage || coreOpts.residKSpace) {
      allData.chip<4>(iv) -= recon->forward(xm);
    }
    if (coreOpts.residImage) {
      xm = recon->adjoint(allData.chip<4>(iv));
      resid.chip<4>(iv) = out_cropper.crop4(xm) / resid.chip<4>(iv).constant(scale);
    }
  }
  Log::Print("All Volumes: {}", Log::ToNow(all_start));
  WriteOutput(coreOpts, out, parser.GetCommand().Name(), traj, Log::Saved(), resid, allData);
  return EXIT_SUCCESS;
}
