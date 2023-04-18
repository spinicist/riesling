#pragma once

#include "tensorop.hpp"

#include "fft/fft.hpp"
#include "tensorOps.hpp"
#include "threads.hpp"

namespace rl {

template <int Rank, int FFTRank>
struct FFTOp final : TensorOperator<Cx, Rank, Rank>
{
  OP_INHERIT(Cx, Rank, Rank)

  FFTOp(InDims const &dims)
    : Parent("FFTOp", dims, dims)
    , fft_{FFT::Make<Rank, FFTRank>(dims)}
  {
  }

  FFTOp(InMap x)
    : Parent("FFTOp", x.dimensions(), x.dimensions())
    , fft_{FFT::Make<Rank, FFTRank>(x)}
  {
  }

  using Parent::adjoint;
  using Parent::forward;

  void forward(InCMap const &x, OutMap &y) const
  {
    auto const time = this->startForward(x);
    y = x;
    fft_->forward(y);
    this->finishForward(y, time);
  }

  void adjoint(OutCMap const &y, InMap &x) const
  {
    auto const time = this->startAdjoint(y);
    x = y;
    fft_->reverse(x);
    this->finishAdjoint(x, time);
  }

private:
  InDims dims_;
  std::shared_ptr<FFT::FFT<Rank, FFTRank>> fft_;
};
} // namespace rl
