#include "fft.hpp"
#include "../fft.hpp"

namespace rl::TOps {

template <int Rank, int FFTRank>
FFT<Rank, FFTRank>::FFT(InDims const &dims)
  : Parent("FFT", dims, dims)
{
  std::iota(dims_.begin(), dims_.end(), Rank - FFTRank);
  ph_ = rl::FFT::PhaseShift(LastN<FFTRank>(ishape));
}

template <int Rank, int FFTRank>
FFT<Rank, FFTRank>::FFT(InMap x)
  : Parent("FFT", x.dimensions(), x.dimensions())
{
  std::iota(dims_.begin(), dims_.end(), Rank - FFTRank);
}

template <int Rank, int FFTRank> void FFT<Rank, FFTRank>::forward(InCMap const &x, OutMap &y) const
{
  auto const time = this->startForward(x, y);
  y = x;
  rl::FFT::Forward(y, dims_, ph_);
  this->finishForward(y, time);
}

template <int Rank, int FFTRank> void FFT<Rank, FFTRank>::adjoint(OutCMap const &y, InMap &x) const
{
  auto const time = this->startAdjoint(y, x);
  x = y;
  rl::FFT::Adjoint(x, dims_, ph_);
  this->finishAdjoint(x, time);
}

template <int Rank, int FFTRank> void FFT<Rank, FFTRank>::iforward(InCMap const &x, OutMap &y) const
{
  auto const time = this->startForward(x, y);
  InTensor   tmp = x;
  rl::FFT::Forward(tmp, dims_, ph_);
  y += tmp;
  this->finishForward(y, time);
}

template <int Rank, int FFTRank> void FFT<Rank, FFTRank>::iadjoint(OutCMap const &y, InMap &x) const
{
  auto const time = this->startAdjoint(y, x);
  InTensor   tmp = y;
  rl::FFT::Adjoint(tmp, dims_, ph_);
  x += tmp;
  this->finishAdjoint(x, time);
}

template struct FFT<4, 3>;
template struct FFT<5, 3>;

} // namespace rl::TOps
