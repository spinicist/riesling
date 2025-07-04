#include "regularizers.hpp"

#include "rl/log/log.hpp"
#include "rl/op/fft.hpp"
#include "rl/op/grad.hpp"
#include "rl/op/laplacian.hpp"
#include "rl/prox/l1-wavelets.hpp"
#include "rl/prox/llr.hpp"
#include "rl/prox/norms.hpp"
#include "rl/algo/eig.hpp"

namespace rl {

RegOpts::RegOpts(args::Subparser &parser)
  : iso(parser, "ISO", "Isotropic/joint dims (b/g/bg)", {"iso"})

  , l1(parser, "L1", "Simple L1 regularization", {"l1"})
  , lap(parser, "L", "Laplacian regularization", {"lap", 'l'})

  , tv(parser, "TV", "Total Variation", {"tv"})
  , tgv(parser, "TGV", "Total Generalized Variation", {"tgv"})

  , tvt(parser, "TVT", "Total Variation along basis dimension", {"tvt"})

  , llr(parser, "L", "LLR regularization", {"llr"})
  , llrPatch(parser, "S", "Patch size for LLR (default 5)", {"llr-patch"}, 5)
  , llrWin(parser, "S", "Window size for LLR (default 3)", {"llr-win"}, 3)
  , llrShift(parser, "S", "Enable random LLR shifting", {"llr-shift"})

  , wavelets(parser, "L", "L1 Wavelet denoising", {"wavelets"})
  , waveDims(parser, "W", "Wavelet transform dimensions (0,1,2)", {"wavelet-dims"}, std::vector<Index>{0, 1, 2})
  , waveWidth(parser, "W", "Wavelet width (4/6/8)", {"wavelet-width"}, 6)
{
}

auto Regularizers(RegOpts &opts, TOps::TOp<Cx, 5, 5>::Ptr const &recon) -> Regularizers_t
{
  Ops::Op<Cx>::Ptr         A = recon;
  auto const               shape = recon->ishape;
  std::vector<Regularizer> regs;

  if (opts.tgv) {
    auto grad_x = TOps::Grad<5>::Make(shape, std::vector<Index>{0, 1, 2});
    auto ext_x = std::make_shared<Ops::Extract<Cx>>(A->cols() + grad_x->rows(), 0, A->cols());
    auto ext_v = std::make_shared<Ops::Extract<Cx>>(A->cols() + grad_x->rows(), A->cols(), grad_x->rows());
    auto op1 = Ops::Sub(Ops::Mul(grad_x, ext_x), ext_v);

    auto grad_v = TOps::GradVec<6>::Make(grad_x->oshape, std::vector<Index>{0, 1, 2});
    auto op2 = Ops::Mul(grad_v, ext_v);

    Proxs::Prox<Cx>::Ptr prox_x, prox_v;
    if (opts.iso) {
      if (opts.iso.Get() == "b") {
        prox_x = std::make_shared<Proxs::L2<6, 1>>(opts.tgv.Get(), grad_x->oshape, Sz1{3});
        prox_v = std::make_shared<Proxs::L2<6, 1>>(opts.tgv.Get(), grad_v->oshape, Sz1{3});
      } else if (opts.iso.Get() == "g") {
        prox_x = std::make_shared<Proxs::L2<6, 1>>(opts.tgv.Get(), grad_x->oshape, Sz1{5});
        prox_v = std::make_shared<Proxs::L2<6, 1>>(opts.tgv.Get(), grad_v->oshape, Sz1{5});
      } else if (opts.iso.Get() == "bg") {
        prox_x = std::make_shared<Proxs::L2<6, 2>>(opts.tgv.Get(), grad_x->oshape, Sz2{3, 5});
        prox_v = std::make_shared<Proxs::L2<6, 2>>(opts.tgv.Get(), grad_v->oshape, Sz2{3, 5});
      } else {
        throw Log::Failure("Regs", "Isotropic dims must be b, g, or bg");
      }
    } else {
      prox_x = std::make_shared<Proxs::L1>(opts.tgv.Get(), op1->rows());
      prox_v = std::make_shared<Proxs::L1>(opts.tgv.Get(), op2->rows());
    }
    regs.push_back({op1, prox_x, grad_x->oshape});
    regs.push_back({op2, prox_v, grad_v->oshape});
    A = std::make_shared<Ops::Multiply<Cx>>(A, ext_x);
    return {regs, A, ext_x};
  }

  if (opts.tv) {
    auto grad = std::make_shared<TOps::Grad<5>>(shape, std::vector<Index>{0, 1, 2});
    Proxs::Prox<Cx>::Ptr prox;
    if (opts.iso) {
      if (opts.iso.Get() == "b") {
        prox = std::make_shared<Proxs::L2<6, 1>>(opts.tv.Get(), grad->oshape, Sz1{3});
      } else if (opts.iso.Get() == "g") {
        prox = std::make_shared<Proxs::L2<6, 1>>(opts.tv.Get(), grad->oshape, Sz1{5});
      } else if (opts.iso.Get() == "bg") {
        prox = std::make_shared<Proxs::L2<6, 2>>(opts.tv.Get(), grad->oshape, Sz2{3, 5});
      } else {
        throw Log::Failure("Regs", "Isotropic dims must be b, g, or bg");
      }
    } else {
      prox = std::make_shared<Proxs::L1>(opts.tv.Get(), grad->rows());
    }
    regs.push_back({grad, prox, grad->oshape});
  }

  if (opts.tvt) {
    auto grad = std::make_shared<TOps::Grad<5>>(shape, std::vector<Index>{0});
    auto prox = std::make_shared<Proxs::L1>(opts.tvt.Get(), grad->rows());
    regs.push_back({grad, prox, grad->oshape});
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

  if (opts.lap) {
    auto lap = std::make_shared<TOps::Laplacian<5>>(shape);
    Proxs::Prox<Cx>::Ptr prox;
    if (opts.iso) {
      if (opts.iso.Get() == "b") {
        prox = std::make_shared<Proxs::L2<5, 1>>(opts.l1.Get(), shape, Sz1{4});
      } else if (opts.iso.Get() == "t") {
        prox = std::make_shared<Proxs::L2<5, 1>>(opts.l1.Get(), shape, Sz1{5});
      } else if (opts.iso.Get() == "bt") {
        prox = std::make_shared<Proxs::L2<5, 2>>(opts.l1.Get(), shape, Sz2{4, 5});
      } else {
        throw Log::Failure("Regs", "Isotropic dims must be b, g, or bg");
      }
    } else {
      prox = std::make_shared<Proxs::L1>(opts.lap.Get(), A->cols());
    }    
    regs.push_back({lap, prox, shape});
  }

  if (opts.l1) {
    Proxs::Prox<Cx>::Ptr prox;
    if (opts.iso) {
      if (opts.iso.Get() == "b") {
        prox = std::make_shared<Proxs::L2<5, 1>>(opts.l1.Get(), shape, Sz1{4});
      } else if (opts.iso.Get() == "t") {
        prox = std::make_shared<Proxs::L2<5, 1>>(opts.l1.Get(), shape, Sz1{5});
      } else if (opts.iso.Get() == "bt") {
        prox = std::make_shared<Proxs::L2<5, 2>>(opts.l1.Get(), shape, Sz2{4, 5});
      } else {
        throw Log::Failure("Regs", "Isotropic dims must be b, g, or bg");
      }
    } else {
      prox = std::make_shared<Proxs::L1>(opts.l1.Get(), A->cols());
    }    
    regs.push_back({nullptr, prox, shape});
  }

  if (regs.size() == 0) { throw Log::Failure("Reg", "Must specify at least one regularizer"); }

  return {regs, A, nullptr};
}

} // namespace rl
