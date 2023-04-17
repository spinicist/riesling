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

  FFTOp(InputMap x)
    : Parent("FFTOp", x.dimensions(), x.dimensions())
    , fft_{FFT::Make<Rank, FFTRank>(x)}
  {
  }

  void forward(InCMap &x, OutMap y) const
  {
    auto const time = this->startForward(x);
    fft_->forward(x);
    this->finishForward(y, time);
  }

  void adjoint(OutCMap &y, InMap x) const
  {
    auto const time = this->startAdjoint(y);
    fft_->reverse(y);
    this->finishAdjoint(x, time);
  }

private:
  InDims dims_;
  std::shared_ptr<FFT::FFT<Rank, FFTRank>> fft_;
};
} // namespace rl
