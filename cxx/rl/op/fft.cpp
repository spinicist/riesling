#include "fft.hpp"

#include "../fft.hpp"
#include "top-impl.hpp"

namespace rl::TOps {

template <int Rank, int FFTRank> FFT<Rank, FFTRank>::FFT(InDims const &shape, bool const adj)
  : Parent(fmt::format("FFT{}", adj ? " Inverse" : ""), shape, shape)
  , adjoint_{adj}
{
  std::iota(dims_.begin(), dims_.end(), 0);
}

template <int Rank, int FFTRank> FFT<Rank, FFTRank>::FFT(InDims const &shape, Sz<FFTRank> const dims, bool const adj)
  : Parent(fmt::format("FFT{}", adj ? " Inverse" : ""), shape, shape)
  , dims_{dims}
  , adjoint_{adj}
{
}

template <int Rank, int FFTRank> FFT<Rank, FFTRank>::FFT(InMap x)
  : Parent("FFT", x.dimensions(), x.dimensions())
{
  std::iota(dims_.begin(), dims_.end(), Rank - FFTRank);
}

template <int Rank, int FFTRank> auto FFT<Rank, FFTRank>::Make(InDims const &shape, bool const adj) -> Ptr
{
  return std::make_shared<FFT<Rank, FFTRank>>(shape, adj);
}

template <int Rank, int FFTRank> void FFT<Rank, FFTRank>::forward(InCMap x, OutMap y, float const s) const
{
  auto const time = this->startForward(x, y, false);
  y.device(Threads::TensorDevice()) = x * x.constant(s);
  if (adjoint_) {
    rl::FFT::Adjoint(y, dims_);
  } else {
    rl::FFT::Forward(y, dims_);
  }
  this->finishForward(y, time, false);
}

template <int Rank, int FFTRank> void FFT<Rank, FFTRank>::adjoint(OutCMap y, InMap x, float const s) const
{
  auto const time = this->startAdjoint(y, x, false);
  x.device(Threads::TensorDevice()) = y * y.constant(s);
  if (adjoint_) {
    rl::FFT::Forward(x, dims_);
  } else {
    rl::FFT::Adjoint(x, dims_);
  }
  this->finishAdjoint(x, time, false);
}

template <int Rank, int FFTRank> void FFT<Rank, FFTRank>::iforward(InCMap x, OutMap y, float const s) const
{
  auto const time = this->startForward(x, y, true);
  InTensor   tmp(x.dimensions());
  tmp.device(Threads::TensorDevice()) = x * x.constant(s);
  if (adjoint_) {
    rl::FFT::Adjoint(tmp, dims_);
  } else {
    rl::FFT::Forward(tmp, dims_);
  }
  y.device(Threads::TensorDevice()) += tmp;
  this->finishForward(y, time, true);
}

template <int Rank, int FFTRank> void FFT<Rank, FFTRank>::iadjoint(OutCMap y, InMap x, float const s) const
{
  auto const time = this->startAdjoint(y, x, true);
  InTensor   tmp(y.dimensions());
  tmp.device(Threads::TensorDevice()) = y * y.constant(s);
  if (adjoint_) {
    rl::FFT::Forward(tmp, dims_);
  } else {
    rl::FFT::Adjoint(tmp, dims_);
  }
  x.device(Threads::TensorDevice()) += tmp;
  this->finishAdjoint(x, time, true);
}

template struct FFT<4, 2>;
template struct FFT<4, 3>;
template struct FFT<5, 1>;
template struct FFT<5, 2>;
template struct FFT<5, 3>;

} // namespace rl::TOps
