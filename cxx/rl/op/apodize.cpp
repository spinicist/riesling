#include "apodize.hpp"

#include "top-impl.hpp"

#include "../fft.hpp"
#include "../kernel/kernel.hpp"
#include "../log/log.hpp"
#include "../tensors.hpp"
#include "pad.hpp"

namespace rl::TOps {

namespace {

template <int ND, typename KF> auto KernelFFT(Sz<ND> const shape, Sz<ND> const gridshape, float const osamp) -> CxN<ND>
{
  Kernel<ND, KF> kernel(osamp);
  CxN<ND>        k = kernel().template cast<Cx>();
  float const    scale = std::sqrt(static_cast<float>(Product(shape)));
  Log::Print("Apodiz", "Shape {} Grid shape {} Scale {}", shape, gridshape, scale);
  k = k * k.constant(scale);
  CxN<ND> temp = TOps::Pad<ND>(k.dimensions(), gridshape).forward(k);
  FFT::Adjoint(temp);
  ReN<ND> a = TOps::Pad<ND>(shape, temp.dimensions()).adjoint(temp).abs().real().cwiseMax(1.e-3f).inverse();
  // if constexpr (ND == 3) { Log::Tensor("apodiz", a.dimensions(), a.data(), HD5::DNames<3>{"i", "j", "k"}); }
  return a.template cast<Cx>();
}
} // namespace

template <int ND, int ED> Apodize<ND, ED, ExpSemi<4>>::Apodize(Sz<ND + ED> const ish, Sz<ND + ED> const osh, float const osamp)
  : Parent("Apodiz", ish, osh)
{
  // Calculate apodization correction
  auto apo_shape = ishape;
  apoBrd_.fill(1);
  for (int ii = 0; ii < ED; ii++) {
    apo_shape[ND + ii] = 1;
    apoBrd_[ND + ii] = ishape[ND + ii];
  }
  apo_ = KernelFFT<ND, ExpSemi<4>>(FirstN<ND>(ishape), FirstN<ND>(oshape), osamp).reshape(apo_shape); // Padding stuff
  Sz<InRank> padRight;
  padLeft_.fill(0);
  padRight.fill(0);
  for (int ii = 0; ii < ND; ii++) {
    padLeft_[ii] = (oshape[ii] - ishape[ii] + 1) / 2;
    padRight[ii] = (oshape[ii] - ishape[ii]) / 2;
  }
  std::transform(padLeft_.cbegin(), padLeft_.cend(), padRight.cbegin(), paddings_.begin(),
                 [](Index left, Index right) { return std::make_pair(left, right); });
}

template <int ND, int ED> void Apodize<ND, ED, ExpSemi<4>>::forward(InCMap x, OutMap y, float const s) const
{
  auto const time = this->startForward(x, y, false);
  y.device(Threads::TensorDevice()) = (x * apo_.broadcast(apoBrd_)).pad(paddings_) * y.constant(s);
  this->finishForward(y, time, false);
}

template <int ND, int ED> void Apodize<ND, ED, ExpSemi<4>>::adjoint(OutCMap y, InMap x, float const s) const
{
  auto const time = this->startAdjoint(y, x, false);
  x.device(Threads::TensorDevice()) = y.slice(padLeft_, ishape) * apo_.broadcast(apoBrd_) * x.constant(s);
  this->finishAdjoint(x, time, false);
}

template <int ND, int ED> void Apodize<ND, ED, ExpSemi<4>>::iforward(InCMap x, OutMap y, float const s) const
{
  auto const time = this->startForward(x, y, true);
  y.device(Threads::TensorDevice()) = (x * apo_.broadcast(apoBrd_)).pad(paddings_) * y.constant(s);
  this->finishForward(y, time, true);
}

template <int ND, int ED> void Apodize<ND, ED, ExpSemi<4>>::iadjoint(OutCMap y, InMap x, float const s) const
{
  auto const time = this->startAdjoint(y, x, true);
  x.device(Threads::TensorDevice()) += y.slice(padLeft_, ishape) * apo_.broadcast(apoBrd_) * x.constant(s) * x.constant(s);
  this->finishAdjoint(x, time, true);
}

template struct Apodize<1, 1, ExpSemi<4>>;
template struct Apodize<2, 1, ExpSemi<4>>;
template struct Apodize<3, 1, ExpSemi<4>>;

template struct Apodize<1, 2, ExpSemi<4>>;
template struct Apodize<2, 2, ExpSemi<4>>;
template struct Apodize<3, 2, ExpSemi<4>>;

template <int ND, int ED> Apodize<ND, ED, TopHat<1>>::Apodize(Sz<ND + ED> const ish, Sz<ND + ED> const osh, float const osamp)
  : Parent("Apodiz", ish, osh)
{
  // Calculate apodization correction
  auto apo_shape = ishape;
  apoBrd_.fill(1);
  for (int ii = 0; ii < ED; ii++) {
    apo_shape[ND + ii] = 1;
    apoBrd_[ND + ii] = ishape[ND + ii];
  }
  Sz<InRank> padRight;
  padLeft_.fill(0);
  padRight.fill(0);
  for (int ii = 0; ii < ND; ii++) {
    padLeft_[ii] = (ishape[ii] - ishape[ii] + 1) / 2;
    padRight[ii] = (ishape[ii] - ishape[ii]) / 2;
  }
  std::transform(padLeft_.cbegin(), padLeft_.cend(), padRight.cbegin(), paddings_.begin(),
                 [](Index left, Index right) { return std::make_pair(left, right); });
}

template <int ND, int ED> void Apodize<ND, ED, TopHat<1>>::forward(InCMap x, OutMap y, float const s) const
{
  auto const time = this->startForward(x, y, false);
  y.device(Threads::TensorDevice()) = x.pad(paddings_) * y.constant(s);
  this->finishForward(y, time, false);
}

template <int ND, int ED> void Apodize<ND, ED, TopHat<1>>::adjoint(OutCMap y, InMap x, float const s) const
{
  auto const time = this->startAdjoint(y, x, false);
  x.device(Threads::TensorDevice()) = y.slice(padLeft_, ishape) * x.constant(s);
  this->finishAdjoint(x, time, false);
}

template <int ND, int ED> void Apodize<ND, ED, TopHat<1>>::iforward(InCMap x, OutMap y, float const s) const
{
  auto const time = this->startForward(x, y, true);
  y.device(Threads::TensorDevice()) = x.pad(paddings_) * y.constant(s);
  this->finishForward(y, time, true);
}

template <int ND, int ED> void Apodize<ND, ED, TopHat<1>>::iadjoint(OutCMap y, InMap x, float const s) const
{
  auto const time = this->startAdjoint(y, x, true);
  x.device(Threads::TensorDevice()) += y.slice(padLeft_, ishape) * x.constant(s);
  this->finishAdjoint(x, time, true);
}

template struct Apodize<1, 1, TopHat<1>>;
template struct Apodize<2, 1, TopHat<1>>;
template struct Apodize<3, 1, TopHat<1>>;

template struct Apodize<1, 2, TopHat<1>>;
template struct Apodize<2, 2, TopHat<1>>;
template struct Apodize<3, 2, TopHat<1>>;

} // namespace rl::TOps
