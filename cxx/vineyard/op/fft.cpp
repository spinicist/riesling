#include "fft.hpp"
#include "../fft.hpp"

namespace rl::Ops {

template <int Rank, int FFTRank>
FFTOp<Rank, FFTRank>::FFTOp(InDims const &dims)
  : Parent("FFTOp", dims, dims)
{
  std::iota(dims_.begin(), dims_.end(), Rank - FFTRank);
  ph_ = FFT::PhaseShift(LastN<FFTRank>(ishape));
  rsh_.fill(1);
  brd_.fill(1);
  std::copy_n(ishape.rbegin(), FFTRank, rsh_.rbegin());
  std::copy_n(ishape.begin(), Rank - FFTRank, brd_.begin());
}

template <int Rank, int FFTRank>
FFTOp<Rank, FFTRank>::FFTOp(InMap x)
  : Parent("FFTOp", x.dimensions(), x.dimensions())
{
  std::iota(dims_.begin(), dims_.end(), Rank - FFTRank);
  ph_ = FFT::PhaseShift(LastN<FFTRank>(ishape));
  rsh_.fill(1);
  brd_.fill(1);
  std::copy_n(ishape.rbegin(), FFTRank, rsh_.rbegin());
  std::copy_n(ishape.begin(), Rank - FFTRank, brd_.begin());
}

template <int Rank, int FFTRank>
void FFTOp<Rank, FFTRank>::forward(InCMap const &x, OutMap &y) const
{
  auto const time = this->startForward(x);
  Log::Debug("y {} x {} ph {} rsh {} brd {}", y.dimensions(), x.dimensions(), ph_.dimensions(), rsh_, brd_);
  y.device(Threads::GlobalDevice()) = x * ph_.reshape(rsh_).broadcast(brd_);
  FFT::Forward(y, dims_);
  y.device(Threads::GlobalDevice()) = y * ph_.reshape(rsh_).broadcast(brd_);
  this->finishForward(y, time);
}

template <int Rank, int FFTRank>
void FFTOp<Rank, FFTRank>::adjoint(OutCMap const &y, InMap &x) const
{
  auto const time = this->startAdjoint(y);
  x.device(Threads::GlobalDevice()) = y / ph_.reshape(rsh_).broadcast(brd_);
  FFT::Adjoint(x, dims_);
  x.device(Threads::GlobalDevice()) = x / ph_.reshape(rsh_).broadcast(brd_);
  this->finishAdjoint(x, time);
}

template struct FFTOp<4, 3>;
template struct FFTOp<5, 3>;

} // namespace rl::Ops
