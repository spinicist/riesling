#include "regularizers.hpp"

#include "rl/algo/eig.hpp"
#include "rl/log/log.hpp"
#include "rl/op/fft.hpp"
#include "rl/op/grad.hpp"
#include "rl/op/laplacian.hpp"
#include "rl/prox/l1-wavelets.hpp"
#include "rl/prox/llr.hpp"
#include "rl/prox/norms.hpp"
#include "rl/prox/stack.hpp"

namespace rl {

RegOpts::RegOpts(args::Subparser &parser)
  : iso(parser, "ISO", "Isotropic/joint dims (b/g/bg)", {"iso"})

  , l1(parser, "L1", "Simple L1 regularization", {"l1"})
  , l1i(parser, "L1", "L1 regularization of the imaginary part", {"l1i"})
  , tv(parser, "TV", "Total Variation", {"tv"})
  , tv2(parser, "TV", "TV + Laplacian", {"tv2"})
  , tgv(parser, "TGV", "Total Generalized Variation", {"tgv"})
  , tvt(parser, "TVT", "Total Variation along time dimension", {"tvt"})

  , llr(parser, "L", "LLR regularization", {"llr"})
  , llrPatch(parser, "S", "Patch size for LLR (default 5)", {"llr-patch"}, 5)
  , llrWin(parser, "S", "Window size for LLR (default 3)", {"llr-win"}, 3)
  , llrShift(parser, "S", "Enable random LLR shifting", {"llr-shift"})

  , wavelets(parser, "L", "L1 Wavelet denoising", {"wavelets"})
  , waveDims(parser, "W", "Wavelet transform dimensions (0,1,2)", {"wavelet-dims"}, std::vector<Index>{0, 1, 2})
  , waveWidth(parser, "W", "Wavelet width (4/6/8)", {"wavelet-width"}, 6)
{
}

auto Regularizers(RegOpts &opts, TOps::TOp<5, 5>::Ptr const &recon) -> Regularizers_t
{
  Ops::Op::Ptr             A = recon;
  auto const               shape = recon->ishape;
  std::vector<Regularizer> regs;

  if (opts.tgv) {
    auto grad_x = TOps::Grad<5, 3>::Make(shape, Sz3{0, 1, 2});
    auto ext_x = std::make_shared<Ops::Extract>(A->cols() + grad_x->rows(), 0, A->cols());
    auto ext_v = std::make_shared<Ops::Extract>(A->cols() + grad_x->rows(), A->cols(), grad_x->rows());
    auto op1 = Ops::Sub(Ops::Mul(grad_x, ext_x), ext_v);

    auto grad_v = TOps::GradVec<6, 3>::Make(grad_x->oshape, Sz3{0, 1, 2});
    auto op2 = Ops::Mul(grad_v, ext_v);

    Proxs::Prox::Ptr prox_x, prox_v;
    if (opts.iso) {
      if (opts.iso.Get() == "b") {
        prox_x = Proxs::L2<6, 1>::Make(opts.tgv.Get(), grad_x->oshape, Sz1{3});
        prox_v = Proxs::L2<6, 1>::Make(opts.tgv.Get(), grad_v->oshape, Sz1{3});
      } else if (opts.iso.Get() == "g") {
        prox_x = Proxs::L2<6, 1>::Make(opts.tgv.Get(), grad_x->oshape, Sz1{5});
        prox_v = Proxs::L2<6, 1>::Make(opts.tgv.Get(), grad_v->oshape, Sz1{5});
      } else if (opts.iso.Get() == "bg") {
        prox_x = Proxs::L2<6, 2>::Make(opts.tgv.Get(), grad_x->oshape, Sz2{3, 5});
        prox_v = Proxs::L2<6, 2>::Make(opts.tgv.Get(), grad_v->oshape, Sz2{3, 5});
      } else {
        throw Log::Failure("Regs", "Isotropic dims must be b, g, or bg");
      }
    } else {
      prox_x = Proxs::L1::Make(opts.tgv.Get(), op1->rows());
      prox_v = Proxs::L1::Make(opts.tgv.Get(), op2->rows());
    }
    regs.push_back({op1, prox_x, grad_x->oshape});
    regs.push_back({op2, prox_v, grad_v->oshape});
    A = std::make_shared<Ops::Multiply>(A, ext_x);
    return {regs, A, ext_x};
  }

  if (opts.tv) {
    auto             grad = TOps::Grad<5, 3>::Make(shape, Sz3{0, 1, 2});
    Proxs::Prox::Ptr prox;
    if (opts.iso) {
      if (opts.iso.Get() == "b") {
        prox = Proxs::L2<6, 1>::Make(opts.tv.Get(), grad->oshape, Sz1{3});
      } else if (opts.iso.Get() == "t") {
        prox = Proxs::L2<6, 1>::Make(opts.tv.Get(), grad->oshape, Sz1{4});
      } else if (opts.iso.Get() == "g") {
        prox = Proxs::L2<6, 1>::Make(opts.tv.Get(), grad->oshape, Sz1{5});
      } else if (opts.iso.Get() == "bt") {
        prox = Proxs::L2<6, 2>::Make(opts.tv.Get(), grad->oshape, Sz2{3, 4});
      } else if (opts.iso.Get() == "bg") {
        prox = Proxs::L2<6, 2>::Make(opts.tv.Get(), grad->oshape, Sz2{3, 5});
      } else if (opts.iso.Get() == "gt") {
        prox = Proxs::L2<6, 2>::Make(opts.tv.Get(), grad->oshape, Sz2{4, 5});
      } else if (opts.iso.Get() == "bgt") {
        prox = Proxs::L2<6, 3>::Make(opts.tv.Get(), grad->oshape, Sz3{3, 4, 5});
      } else {
        throw Log::Failure("Regs", "Valid dims are bgt");
      }
    } else {
      prox = Proxs::L1::Make(opts.tv.Get(), grad->rows());
    }
    regs.push_back({grad, prox, grad->oshape});
  }

  if (opts.tv2) {
    auto grad = TOps::Grad<5, 3>::Make(shape, Sz3{0, 1, 2});
    auto lap = TOps::IsoΔ3D<5>::Make(shape);
    auto both = Ops::VStack::Make({grad, lap});

    Proxs::Prox::Ptr pg, pl;
    float const      σ = 0.77; // Bock et al 2008
    if (opts.iso) {
      if (opts.iso.Get() == "b") {
        pg = Proxs::L2<6, 1>::Make(opts.tv2.Get() * σ, grad->oshape, Sz1{3});
        pl = Proxs::L2<5, 1>::Make(opts.tv2.Get() * (1 - σ), lap->oshape, Sz1{3});
      } else if (opts.iso.Get() == "t") {
        pg = Proxs::L2<6, 1>::Make(opts.tv2.Get() * σ, grad->oshape, Sz1{4});
        pl = Proxs::L2<5, 1>::Make(opts.tv2.Get() * (1 - σ), lap->oshape, Sz1{4});
      } else if (opts.iso.Get() == "bt") {
        pg = Proxs::L2<6, 2>::Make(opts.tv2.Get() * σ, grad->oshape, Sz2{3, 4});
        pl = Proxs::L2<5, 2>::Make(opts.tv2.Get() * (1 - σ), lap->oshape, Sz2{3, 4});
      } else {
        throw Log::Failure("Regs", "Isotropic dims must bebt");
      }
    } else {
      pg = Proxs::L1::Make(opts.tv2.Get() * σ, grad->rows());
      pl = Proxs::L1::Make(opts.tv2.Get() * (1 - σ), lap->rows());
    }
    auto prox = Proxs::Stack::Make({pg, pl});
    regs.push_back({both, prox, grad->oshape});
  }

  if (opts.tvt) {
    auto grad = TOps::Grad<5, 1>::Make(shape, Sz1{4});
    auto prox = Proxs::L1::Make(opts.tvt.Get(), grad->rows());
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

  if (opts.l1) {
    Proxs::Prox::Ptr prox;
    if (opts.iso) {
      if (opts.iso.Get() == "b") {
        prox = std::make_shared<Proxs::L2<5, 1>>(opts.l1.Get(), shape, Sz1{4});
      } else if (opts.iso.Get() == "t") {
        prox = std::make_shared<Proxs::L2<5, 1>>(opts.l1.Get(), shape, Sz1{5});
      } else if (opts.iso.Get() == "bt") {
        prox = std::make_shared<Proxs::L2<5, 2>>(opts.l1.Get(), shape, Sz2{4, 5});
      } else {
        throw Log::Failure("Regs", "Valid dims are bt");
      }
    } else {
      prox = Proxs::L1::Make(opts.l1.Get(), A->cols());
    }
    regs.push_back({nullptr, prox, shape});
  }

  if (opts.l1i) {
    Proxs::Prox::Ptr prox = Proxs::L1I::Make(opts.l1i.Get(), A->cols());
    regs.push_back({nullptr, prox, shape});
  }

  if (regs.size() == 0) { throw Log::Failure("Reg", "Must specify at least one regularizer"); }

  return {regs, A, nullptr};
}

} // namespace rl
