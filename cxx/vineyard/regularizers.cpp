#include "regularizers.hpp"

#include "algo/eig.hpp"
#include "op/fft.hpp"
#include "op/grad.hpp"
#include "prox/entropy.hpp"
#include "prox/l1-wavelets.hpp"
#include "prox/llr.hpp"
#include "prox/norms.hpp"

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

  , wavelets(parser, "L", "L1 Wavelet denoising", {"wavelets"})
  , waveDims(parser, "W", "Wavelet transform dimensions (b,x,y,z 0/1)", {"wavelet-dims"}, Sz4{0, 1, 1, 1})
  , waveWidth(parser, "W", "Wavelet width (4/6/8)", {"wavelet-width"}, 6)
{
}

Regularizers::Regularizers(RegOpts &opts, Sz5 const shape, std::shared_ptr<Ops::Op<Cx>> &A)
{
  ext_x = std::make_shared<TOps::Identity<Cx, 5>>(shape); // Need for TGV, sigh

  if (opts.tgv) {
    if (opts.tgvl2) { Log::Fail("You tried to TGVL2 your TGV. Nope."); }
    auto grad_x = std::make_shared<TOps::Grad>(shape, std::vector<Index>{1, 2, 3});
    ext_x = std::make_shared<Ops::Extract<Cx>>(A->cols() + grad_x->rows(), 0, A->cols());
    auto ext_v = std::make_shared<Ops::Extract<Cx>>(A->cols() + grad_x->rows(), A->cols(), grad_x->rows());
    auto op1 = std::make_shared<Ops::Subtract<Cx>>(std::make_shared<Ops::Multiply<Cx>>(grad_x, ext_x), ext_v);
    auto prox1 = std::make_shared<Proxs::L1>(opts.tgv.Get(), op1->rows());
    auto grad_v = std::make_shared<TOps::GradVec>(grad_x->oshape);
    auto op2 = std::make_shared<Ops::Multiply<Cx>>(grad_v, ext_v);
    auto prox2 = std::make_shared<Proxs::L1>(opts.tgv.Get(), op2->rows());
    regs.push_back({op1, prox1});
    regs.push_back({op2, prox2});
    sizes = {grad_x->oshape, grad_v->oshape};
    A = std::make_shared<Ops::Multiply<Cx>>(A, ext_x);
  }

  if (opts.tgvl2) {
    if (opts.tgv) { Log::Fail("You tried to TGV your TGV-L2. Nope."); }
    auto grad_x = std::make_shared<TOps::Grad>(shape, std::vector<Index>{1, 2, 3});
    ext_x = std::make_shared<Ops::Extract<Cx>>(A->cols() + grad_x->rows(), 0, A->cols());
    auto ext_v = std::make_shared<Ops::Extract<Cx>>(A->cols() + grad_x->rows(), A->cols(), grad_x->rows());
    auto op1 = std::make_shared<Ops::Subtract<Cx>>(std::make_shared<Ops::Multiply<Cx>>(grad_x, ext_x), ext_v);
    auto prox1 = std::make_shared<Proxs::L2>(opts.tgvl2.Get(), op1->rows(), shape[0]);
    auto grad_v = std::make_shared<TOps::GradVec>(grad_x->oshape);
    auto op2 = std::make_shared<Ops::Multiply<Cx>>(grad_v, ext_v);
    auto prox2 = std::make_shared<Proxs::L2>(opts.tgvl2.Get(), op2->rows(), shape[0]);
    regs.push_back({op1, prox1});
    regs.push_back({op2, prox2});
    sizes = {grad_x->oshape, grad_v->oshape};
    A = std::make_shared<Ops::Multiply<Cx>>(A, ext_x);
  }

  if (opts.wavelets) {
    regs.push_back(
      {std::make_shared<Ops::Multiply<Cx>>(std::make_shared<TOps::Identity<Cx, 4>>(shape), ext_x),
       std::make_shared<Proxs::L1Wavelets>(opts.wavelets.Get(), shape, opts.waveWidth.Get(), opts.waveDims.Get())});
    sizes.push_back(shape);
  }

  if (opts.llr) {
    regs.push_back(
      {std::make_shared<Ops::Multiply<Cx>>(std::make_shared<TOps::Identity<Cx, 4>>(shape), ext_x),
       std::make_shared<Proxs::LLR>(opts.llr.Get(), opts.llrPatch.Get(), opts.llrWin.Get(), opts.llrShift, shape)});
    sizes.push_back(shape);
  }

  if (opts.l1) {
    auto op = std::make_shared<Ops::Multiply<Cx>>(std::make_shared<TOps::Identity<Cx, 4>>(shape), ext_x);
    auto p = std::make_shared<Proxs::L1>(opts.l1.Get(), op->rows());
    regs.push_back({op, p});
    sizes.push_back(shape);
  }

  if (opts.nmrent) {
    auto op = std::make_shared<Ops::Multiply<Cx>>(std::make_shared<TOps::Identity<Cx, 4>>(shape), ext_x);
    auto p = std::make_shared<Proxs::NMREntropy>(opts.nmrent.Get(), op->rows());
    regs.push_back({op, p});
    sizes.push_back(shape);
  }

  if (opts.tv) {
    auto grad = std::make_shared<TOps::Grad>(shape, std::vector<Index>{1, 2, 3});
    auto op = std::make_shared<Ops::Multiply<Cx>>(grad, ext_x);
    auto prox = std::make_shared<Proxs::L1>(opts.tv.Get(), op->rows());
    regs.push_back({op, prox});
    sizes.push_back(grad->oshape);
  }

  if (opts.tvt) {
    auto grad = std::make_shared<TOps::Grad>(shape, std::vector<Index>{0});
    auto op = std::make_shared<Ops::Multiply<Cx>>(grad, ext_x);
    auto prox = std::make_shared<Proxs::L1>(opts.tvt.Get(), op->rows());
    regs.push_back({op, prox});
    sizes.push_back(grad->oshape);
  }

  if (regs.size() == 0) { Log::Fail("Must specify at least one regularizer"); }
}

} // namespace rl
