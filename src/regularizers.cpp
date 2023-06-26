#include "regularizers.hpp"

#include "algo/eig.hpp"
#include "op/grad.hpp"
#include "prox/entropy.hpp"
#include "prox/llr.hpp"
#include "prox/thresh-wavelets.hpp"

namespace rl {

RegOpts::RegOpts(args::Subparser &parser)
  : tv(parser, "TV", "Total Variation", {"tv"})
  , tvt(parser, "TVT", "Total Variation along time/frames/basis", {"tvt"})
  , tgv(parser, "TGV", "Total Generalized Variation", {"tgv"})
  , l1(parser, "L1", "Simple L1 regularization", {"l1"})
  , nmrent(parser, "E", "NMR Entropy", {"nmrent"})
  ,

  llr(parser, "L", "LLR regularization", {"llr"})
  , llrPatch(parser, "S", "Patch size for LLR (default 4)", {"llr-patch"}, 5)
  , llrWin(parser, "S", "Patch size for LLR (default 4)", {"llr-win"}, 3)
  , llrShift(parser, "S", "Enable random LLR shifting", {"llr-shift"})
  ,

  wavelets(parser, "L", "L1 Wavelet denoising", {"wavelets"})
  , waveLevels(parser, "W", "Wavelet denoising levels", {"wavelet-levels"}, 4)
  , waveWidth(parser, "W", "Wavelet width (4/6/8)", {"wavelet-width"}, 6)
{
}

Regularizers::Regularizers(RegOpts &opts, Sz4 const shape, std::shared_ptr<Ops::Op<Cx>> &A)
{
  ext_x = std::make_shared<TensorIdentity<Cx, 4>>(shape); // Need for TGV, sigh

  if (opts.tgv) {
    auto grad_x = std::make_shared<GradOp>(shape, std::vector<Index>{1, 2, 3});
    ext_x = std::make_shared<Ops::Extract<Cx>>(A->cols() + grad_x->rows(), 0, A->cols());
    auto ext_v = std::make_shared<Ops::Extract<Cx>>(A->cols() + grad_x->rows(), A->cols(), grad_x->rows());
    auto op1 = std::make_shared<Ops::Subtract<Cx>>(std::make_shared<Ops::Multiply<Cx>>(grad_x, ext_x), ext_v);
    auto prox1 = std::make_shared<Proxs::SoftThreshold>(opts.tgv.Get(), op1->rows());
    auto grad_v = std::make_shared<GradVecOp>(grad_x->oshape);
    auto op2 = std::make_shared<Ops::Multiply<Cx>>(grad_v, ext_v);
    auto prox2 = std::make_shared<Proxs::SoftThreshold>(opts.tgv.Get(), op2->rows());
    prox = {prox1, prox2};
    ops = {op1, op2};
    A = std::make_shared<Ops::Multiply<Cx>>(A, ext_x);
  }

  if (opts.wavelets) {
    ops.push_back(std::make_shared<Ops::Multiply<Cx>>(std::make_shared<TensorIdentity<Cx, 4>>(shape), ext_x));
    prox.push_back(
      std::make_shared<Proxs::ThresholdWavelets>(opts.wavelets.Get(), shape, opts.waveWidth.Get(), opts.waveLevels.Get()));
  }

  if (opts.llr) {
    ops.push_back(std::make_shared<Ops::Multiply<Cx>>(std::make_shared<TensorIdentity<Cx, 4>>(shape), ext_x));
    prox.push_back(std::make_shared<Proxs::LLR>(opts.llr.Get(), opts.llrPatch.Get(), opts.llrWin.Get(), opts.llrShift, shape));
  }

  if (opts.nmrent) {
    ops.push_back(std::make_shared<Ops::Multiply<Cx>>(std::make_shared<TensorIdentity<Cx, 4>>(shape), ext_x));
    prox.push_back(std::make_shared<Proxs::NMREntropy>(opts.nmrent.Get(), ops.back()->rows()));
  }

  if (opts.l1) {
    ops.push_back(std::make_shared<Ops::Multiply<Cx>>(std::make_shared<TensorIdentity<Cx, 4>>(shape), ext_x));
    prox.push_back(std::make_shared<Proxs::SoftThreshold>(opts.l1.Get(), ops.back()->rows()));
  }

  if (opts.tv) {
    ops.push_back(std::make_shared<Ops::Multiply<Cx>>(std::make_shared<GradOp>(shape, std::vector<Index>{1, 2, 3}), ext_x));
    prox.push_back(std::make_shared<Proxs::SoftThreshold>(opts.tv.Get(), ops.back()->rows()));
  }

  if (opts.tvt) {
    ops.push_back(std::make_shared<Ops::Multiply<Cx>>(std::make_shared<GradOp>(shape, std::vector<Index>{0}), ext_x));
    prox.push_back(std::make_shared<Proxs::SoftThreshold>(opts.tvt.Get(), ops.back()->rows()));
  }

  if (prox.size() == 0) { Log::Fail("Must specify at least one regularizer"); }
}

auto Regularizers::count() const -> Index { return ops.size(); }

auto Regularizers::σ(std::vector<float> σin) const -> std::vector<float>
{
  if (σin.size() == 0) {
    for (auto &G : ops) {
      auto eigG = PowerMethod(G, 32);
      σin.push_back(1.f / eigG.val);
    }
    return σin;
  } else if (σin.size() == ops.size()) {
    return σin;
  } else {
    Log::Fail("Number of σ {} does not match number of regularizers {}", σin.size(), ops.size());
  }
}

} // namespace rl
