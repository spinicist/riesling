#pragma once

#include "operator.hpp"

#include "fft/fft.hpp"
#include "tensorOps.hpp"
#include "threads.hpp"

namespace rl {

template <int Rank, int FFTRank>
struct FFTOp final : Operator<Cx, Rank, Rank>
{
  OP_INHERIT(Cx, Rank, Rank)

  FFTOp(InputDims const &dims)
    : Parent("FFTOp", dims, dims)
    , fft_{FFT::Make<Rank, FFTRank>(dims)}
  {
  }

  FFTOp(InputMap x)
    : Parent("FFTOp", x.dimensions(), x.dimensions())
    , fft_{FFT::Make<Rank, FFTRank>(x)}
  {
  }

  auto forward(InputMap x) const -> OutputMap
  {
    auto const time = this->startForward(x);
    fft_->forward(x);
    this->finishForward(x, time);
    return x;
  }

  auto adjoint(OutputMap x) const -> InputMap
  {
    auto const time = this->startAdjoint(x);
    fft_->reverse(x);
    this->finishAdjoint(x, time);
    return x;
  }

private:
  InputDims dims_;
  std::shared_ptr<FFT::FFT<Rank, FFTRank>> fft_;
};
} // namespace rl
