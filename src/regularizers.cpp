#include "regularizers.hpp"

#include "algo/eig.hpp"
#include "op/fft.hpp"
#include "op/grad.hpp"
#include "prox/entropy.hpp"
#include "prox/llr.hpp"
#include "prox/norms.hpp"
#include "prox/l1-wavelets.hpp"

namespace rl {

RegOpts::RegOpts(args::Subparser &parser)
  : l1(parser, "L1", "Simple L1 regularization", {"l1"})
  , nmrent(parser, "E", "NMR Entropy", {"nmrent"})

  , tv(parser, "TV", "Total Variation", {"tv"})
  , tvt(parser, "TVT", "Total Variation along time/frames/basis", {"tvt"})

  , tgv(parser, "TGV", "Total Generalized Variation", {"tgv"})
  , tgvl2(parser, "TGV-L2", "TGV with voxel-wise L2 norm", {"tgvl2"})

  , llr(parser, "L", "LLR regularization", {"llr"})
  , llrPatch(parser, "S", "Patch size for LLR (default 5)", {"llr-patch"}, 5)
  , llrWin(parser, "S", "Window size for LLR (default 3)", {"llr-win"}, 3)
  , llrShift(parser, "S", "Enable random LLR shifting", {"llr-shift"})
  , llrFFT(parser, "F", "Perform LLR in the Fourier domain", {"llr-fft"})

  , wavelets(parser, "L", "L1 Wavelet denoising", {"wavelets"})
  , waveDims(parser, "W", "Wavelet transform dimensions (b,x,y,z 0/1)", {"wavelet-dims"}, Sz4{0, 1, 1, 1})
  , waveWidth(parser, "W", "Wavelet width (4/6/8)", {"wavelet-width"}, 6)
{
}

Regularizers::Regularizers(RegOpts &opts, Sz4 const shape, std::shared_ptr<Ops::Op<Cx>> &A)
{
  ext_x = std::make_shared<TensorIdentity<Cx, 4>>(shape); // Need for TGV, sigh

  if (opts.tgv) {
    if (opts.tgvl2) {
      Log::Fail("You tried to TGVL2 your TGV. Nope.");
    }
    auto grad_x = std::make_shared<GradOp>(shape, std::vector<Index>{1, 2, 3});
    ext_x = std::make_shared<Ops::Extract<Cx>>(A->cols() + grad_x->rows(), 0, A->cols());
    auto ext_v = std::make_shared<Ops::Extract<Cx>>(A->cols() + grad_x->rows(), A->cols(), grad_x->rows());
    auto op1 = std::make_shared<Ops::Subtract<Cx>>(std::make_shared<Ops::Multiply<Cx>>(grad_x, ext_x), ext_v);
    auto prox1 = std::make_shared<Proxs::L1>(opts.tgv.Get(), op1->rows());
    auto grad_v = std::make_shared<GradVecOp>(grad_x->oshape);
    auto op2 = std::make_shared<Ops::Multiply<Cx>>(grad_v, ext_v);
    auto prox2 = std::make_shared<Proxs::L1>(opts.tgv.Get(), op2->rows());
    prox = {prox1, prox2};
    ops = {op1, op2};
    sizes = {grad_x->oshape, grad_v->oshape};
    A = std::make_shared<Ops::Multiply<Cx>>(A, ext_x);
  }

  if (opts.tgvl2) {
    if (opts.tgv) {
      Log::Fail("You tried to TGV your TGV-L2. Nope.");
    }
    auto grad_x = std::make_shared<GradOp>(shape, std::vector<Index>{1, 2, 3});
    ext_x = std::make_shared<Ops::Extract<Cx>>(A->cols() + grad_x->rows(), 0, A->cols());
    auto ext_v = std::make_shared<Ops::Extract<Cx>>(A->cols() + grad_x->rows(), A->cols(), grad_x->rows());
    auto op1 = std::make_shared<Ops::Subtract<Cx>>(std::make_shared<Ops::Multiply<Cx>>(grad_x, ext_x), ext_v);
    auto prox1 = std::make_shared<Proxs::L2>(opts.tgvl2.Get(), op1->rows(), shape[0]);
    auto grad_v = std::make_shared<GradVecOp>(grad_x->oshape);
    auto op2 = std::make_shared<Ops::Multiply<Cx>>(grad_v, ext_v);
    auto prox2 = std::make_shared<Proxs::L2>(opts.tgvl2.Get(), op2->rows(), shape[0]);
    prox = {prox1, prox2};
    ops = {op1, op2};
    sizes = {grad_x->oshape, grad_v->oshape};
    A = std::make_shared<Ops::Multiply<Cx>>(A, ext_x);
  }

  if (opts.wavelets) {
    ops.push_back(std::make_shared<Ops::Multiply<Cx>>(std::make_shared<TensorIdentity<Cx, 4>>(shape), ext_x));
    prox.push_back(
      std::make_shared<Proxs::L1Wavelets>(opts.wavelets.Get(), shape, opts.waveWidth.Get(), opts.waveDims.Get()));
    sizes.push_back(shape);
  }

  if (opts.llr) {
    if (opts.llrFFT) {
      ops.push_back(std::make_shared<Ops::Multiply<Cx>>(std::make_shared<Ops::FFTOp<4, 3>>(shape), ext_x));
    } else {
      ops.push_back(std::make_shared<Ops::Multiply<Cx>>(std::make_shared<TensorIdentity<Cx, 4>>(shape), ext_x));
    }
    prox.push_back(std::make_shared<Proxs::LLR>(opts.llr.Get(), opts.llrPatch.Get(), opts.llrWin.Get(), opts.llrShift, shape));
    sizes.push_back(shape);
  }

  if (opts.l1) {
    ops.push_back(std::make_shared<Ops::Multiply<Cx>>(std::make_shared<TensorIdentity<Cx, 4>>(shape), ext_x));
    prox.push_back(std::make_shared<Proxs::L1>(opts.l1.Get(), ops.back()->rows()));
    sizes.push_back(shape);
  }

  if (opts.nmrent) {
    ops.push_back(std::make_shared<Ops::Multiply<Cx>>(std::make_shared<TensorIdentity<Cx, 4>>(shape), ext_x));
    prox.push_back(std::make_shared<Proxs::NMREntropy>(opts.nmrent.Get(), ops.back()->rows()));
    sizes.push_back(shape);
  }

  if (opts.tv) {
    auto grad = std::make_shared<GradOp>(shape, std::vector<Index>{1, 2, 3});
    ops.push_back(std::make_shared<Ops::Multiply<Cx>>(grad, ext_x));
    prox.push_back(std::make_shared<Proxs::L1>(opts.tv.Get(), ops.back()->rows()));
    sizes.push_back(grad->oshape);
  }

  if (opts.tvt) {
    auto grad = std::make_shared<GradOp>(shape, std::vector<Index>{0});
    ops.push_back(std::make_shared<Ops::Multiply<Cx>>(grad, ext_x));
    prox.push_back(std::make_shared<Proxs::L1>(opts.tvt.Get(), ops.back()->rows()));
    sizes.push_back(grad->oshape);
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
