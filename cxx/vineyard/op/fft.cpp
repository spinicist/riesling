#include "fft.hpp"

namespace rl::Ops {

template <int Rank, int FFTRank>
FFTOp<Rank, FFTRank>::FFTOp(InDims const &dims)
  : Parent("FFTOp", dims, dims)
  , fft_{FFT::Make<Rank, FFTRank>(dims)}
{
}

template <int Rank, int FFTRank>
FFTOp<Rank, FFTRank>::FFTOp(InMap x)
  : Parent("FFTOp", x.dimensions(), x.dimensions())
  , fft_{FFT::Make<Rank, FFTRank>(x)}
{
}
template <int Rank, int FFTRank>

void FFTOp<Rank, FFTRank>::forward(InCMap const &x, OutMap &y) const
{
  auto const time = this->startForward(x);
  y = x;
  fft_->forward(y);
  this->finishForward(y, time);
}

template <int Rank, int FFTRank>
void FFTOp<Rank, FFTRank>::adjoint(OutCMap const &y, InMap &x) const
{
  auto const time = this->startAdjoint(y);
  x = y;
  fft_->reverse(x);
  this->finishAdjoint(x, time);
}

template struct FFTOp<4, 3>;
template struct FFTOp<5, 3>;

} // namespace rl::Ops
