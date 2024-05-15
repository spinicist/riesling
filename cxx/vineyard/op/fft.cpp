#include "fft.hpp"
#include "../fft.hpp"

namespace rl::Ops {

template <int Rank, int FFTRank>
FFTOp<Rank, FFTRank>::FFTOp(InDims const &dims)
  : Parent("FFTOp", dims, dims)
{
  std::iota(dims_.begin(), dims_.end(), Rank - FFTRank);
  ph_ = FFT::PhaseShift(LastN<FFTRank>(ishape));
}

template <int Rank, int FFTRank>
FFTOp<Rank, FFTRank>::FFTOp(InMap x)
  : Parent("FFTOp", x.dimensions(), x.dimensions())
{
  std::iota(dims_.begin(), dims_.end(), Rank - FFTRank);
}

template <int Rank, int FFTRank>
void FFTOp<Rank, FFTRank>::forward(InCMap const &x, OutMap &y) const
{
  auto const time = this->startForward(x);
  y = x;
  FFT::Forward(y, dims_, ph_);
  this->finishForward(y, time);
}

template <int Rank, int FFTRank>
void FFTOp<Rank, FFTRank>::adjoint(OutCMap const &y, InMap &x) const
{
  auto const time = this->startAdjoint(y);
  x = y;
  FFT::Adjoint(x, dims_, ph_);
  this->finishAdjoint(x, time);
}

template struct FFTOp<4, 3>;
template struct FFTOp<5, 3>;

} // namespace rl::Ops
