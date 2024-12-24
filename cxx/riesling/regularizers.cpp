#include "regularizers.hpp"

#include "rl/log.hpp"
#include "rl/op/fft.hpp"
#include "rl/op/grad.hpp"
#include "rl/prox/entropy.hpp"
#include "rl/prox/l1-wavelets.hpp"
#include "rl/prox/llr.hpp"
#include "rl/prox/norms.hpp"

namespace rl {

RegOpts::RegOpts(args::Subparser &parser)
  : l1(parser, "L1", "Simple L1 regularization", {"l1"})
  , nmrent(parser, "E", "NMR Entropy", {"nmrent"})

  , diffOrder(parser, "G", "Finite difference scheme", {"diff"}, 0)
  , tv(parser, "TV", "Total Variation", {"tv"})
  , tvl2(parser, "TV-L2", "Total Variation with L2 norm across b", {"tvl2"})
  , tvt(parser, "TVT", "Total Variation along basis dimension", {"tvt"})

  , tgv(parser, "TGV", "Total Generalized Variation", {"tgv"})
  , tgvl2(parser, "TGV-L2", "TGV with L2 norm across b", {"tgvl2"})

  , llr(parser, "L", "LLR regularization", {"llr"})
  , llrPatch(parser, "S", "Patch size for LLR (default 5)", {"llr-patch"}, 5)
  , llrWin(parser, "S", "Window size for LLR (default 3)", {"llr-win"}, 3)
  , llrShift(parser, "S", "Enable random LLR shifting", {"llr-shift"})

  , wavelets(parser, "L", "L1 Wavelet denoising", {"wavelets"})
  , waveDims(parser, "W", "Wavelet transform dimensions (b,x,y,z 0/1)", {"wavelet-dims"}, std::vector<Index>{1, 2, 3})
  , waveWidth(parser, "W", "Wavelet width (4/6/8)", {"wavelet-width"}, 6)
{
}

auto Regularizers(RegOpts &opts, TOps::TOp<Cx, 5, 5>::Ptr const &recon) -> Regularizers_t
{
  Ops::Op<Cx>::Ptr         A = recon;
  auto const               shape = recon->ishape;
  Ops::Op<Cx>::Ptr         ext_x = std::make_shared<TOps::Identity<Cx, 5>>(shape); // Need for TGV, sigh
  std::vector<Regularizer> regs;

  if (opts.tgv) {
    if (opts.tgvl2) { throw Log::Failure("Reg", "You tried to TGVL2 your TGV. Nope."); }
    auto grad_x = std::make_shared<TOps::Grad<5>>(shape, std::vector<Index>{1, 2, 3}, opts.diffOrder.Get());
    ext_x = std::make_shared<Ops::Extract<Cx>>(A->cols() + grad_x->rows(), 0, A->cols());
    auto ext_v = std::make_shared<Ops::Extract<Cx>>(A->cols() + grad_x->rows(), A->cols(), grad_x->rows());
    auto op1 = std::make_shared<Ops::Subtract<Cx>>(std::make_shared<Ops::Multiply<Cx>>(grad_x, ext_x), ext_v);
    auto prox1 = std::make_shared<Proxs::L1>(opts.tgv.Get(), op1->rows());
    auto grad_v = std::make_shared<TOps::GradVec<6>>(grad_x->oshape, std::vector<Index>{1, 2, 3}, opts.diffOrder.Get());
    auto op2 = std::make_shared<Ops::Multiply<Cx>>(grad_v, ext_v);
    auto prox2 = std::make_shared<Proxs::L1>(opts.tgv.Get(), op2->rows());
    regs.push_back({op1, prox1, grad_x->oshape});
    regs.push_back({op2, prox2, grad_v->oshape});
    A = std::make_shared<Ops::Multiply<Cx>>(A, ext_x);
  }

  if (opts.tgvl2) {
    if (opts.tgv) { throw Log::Failure("Reg", "You tried to TGV your TGV-L2. Nope."); }
    auto grad_x = std::make_shared<TOps::Grad<5>>(shape, std::vector<Index>{1, 2, 3}, opts.diffOrder.Get());
    ext_x = std::make_shared<Ops::Extract<Cx>>(A->cols() + grad_x->rows(), 0, A->cols());
    auto ext_v = std::make_shared<Ops::Extract<Cx>>(A->cols() + grad_x->rows(), A->cols(), grad_x->rows());
    auto op1 = std::make_shared<Ops::Subtract<Cx>>(std::make_shared<Ops::Multiply<Cx>>(grad_x, ext_x), ext_v);
    auto prox1 = std::make_shared<Proxs::L2>(opts.tgvl2.Get(), op1->rows(), shape[0]);
    auto grad_v = std::make_shared<TOps::GradVec<6>>(grad_x->oshape, std::vector<Index>{1, 2, 3}, opts.diffOrder.Get());
    auto op2 = std::make_shared<Ops::Multiply<Cx>>(grad_v, ext_v);
    auto prox2 = std::make_shared<Proxs::L2>(opts.tgvl2.Get(), op2->rows(), shape[0]);
    regs.push_back({op1, prox1, grad_x->oshape});
    regs.push_back({op2, prox2, grad_v->oshape});
    A = std::make_shared<Ops::Multiply<Cx>>(A, ext_x);
  }

  if (opts.wavelets) {
    regs.push_back({nullptr,
                    std::make_shared<Proxs::L1Wavelets>(opts.wavelets.Get(), shape, opts.waveWidth.Get(), opts.waveDims.Get()),
                    shape});
  }

  if (opts.llr) {
    regs.push_back({nullptr,
                    std::make_shared<Proxs::LLR>(opts.llr.Get(), opts.llrPatch.Get(), opts.llrWin.Get(), opts.llrShift, shape),
                    shape});
  }

  if (opts.l1) {
    auto p = std::make_shared<Proxs::L1>(opts.l1.Get(), ext_x->rows());
    regs.push_back({nullptr, p, shape});
  }

  if (opts.nmrent) {
    auto p = std::make_shared<Proxs::NMREntropy>(opts.nmrent.Get(), ext_x->rows());
    regs.push_back({nullptr, p, shape});
  }

  if (opts.tv) {
    auto grad = std::make_shared<TOps::Grad<5>>(shape, std::vector<Index>{1, 2, 3}, opts.diffOrder.Get());
    auto op = std::make_shared<Ops::Multiply<Cx>>(grad, ext_x);
    auto prox = std::make_shared<Proxs::L1>(opts.tv.Get(), op->rows());
    regs.push_back({op, prox, grad->oshape});
  }

  if (opts.tvl2) {
    auto grad = std::make_shared<TOps::Grad<5>>(shape, std::vector<Index>{1, 2, 3}, opts.diffOrder.Get());
    auto op = std::make_shared<Ops::Multiply<Cx>>(grad, ext_x);
    auto prox = std::make_shared<Proxs::L2>(opts.tvl2.Get(), op->rows(), shape[0]);
    regs.push_back({op, prox, grad->oshape});
  }

  if (opts.tvt) {
    auto grad = std::make_shared<TOps::Grad<5>>(shape, std::vector<Index>{0}, opts.diffOrder.Get());
    auto op = std::make_shared<Ops::Multiply<Cx>>(grad, ext_x);
    auto prox = std::make_shared<Proxs::L1>(opts.tvt.Get(), op->rows());
    regs.push_back({op, prox, grad->oshape});
  }

  if (regs.size() == 0) { throw Log::Failure("Reg", "Must specify at least one regularizer"); }

  return {regs, A, ext_x};
}

} // namespace rl
