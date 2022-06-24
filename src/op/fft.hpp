#pragma once

#include "operator.hpp"

#include "../fft/fft.hpp"
#include "../threads.h"

template <int Rank, int FFTRank = 3>
struct FFTOp final : Operator<Rank, Rank>
{
  using Parent = Operator<Rank, Rank>;
  using Input = typename Parent::Input;
  using InputDims = typename Parent::InputDims;
  using Output = typename Parent::Output;
  using OutputDims = typename Parent::OutputDims;
  using Tensor = typename Eigen::Tensor<Cx, Rank>;

  FFTOp(InputDims const &dims)
    : dims_{dims}
    , ws_{std::make_shared<Tensor>(dims_)}
    , fft_{FFT::Make<5, 3>(*ws_)}
  {
  }

  FFTOp(std::shared_ptr<Tensor> ws)
    : dims_{ws->dimensions()}
    , ws_{ws}
    , fft_{FFT::Make<5, 3>(*ws_)}
  {
  }

  InputDims inputDimensions() const
  {
    return dims_;
  }

  OutputDims outputDimensions() const
  {
    return dims_;
  }

  template <typename T>
  Tensor const &A(T const &x) const
  {
    Log::Debug("Out-of-place Forward FFT Op");
    ws_->device(Threads::GlobalDevice()) = x;
    fft_->forward(*ws_);
    return *ws_;
  }

  template <typename T>
  Tensor &Adj(T const &x) const
  {
    Log::Debug("Out-of-place Adjoint FFT Op");
    ws_->device(Threads::GlobalDevice()) = x;
    fft_->reverse(*ws_);
    return *ws_;
  }

  Tensor const &A(Tensor &x) const
  {
    Log::Debug("In-place Forward FFT Op");
    fft_->forward(x);
    return x;
  }

  Tensor &Adj(Tensor &x) const
  {
    Log::Debug("In-place Adjoint FFT Op");
    fft_->reverse(x);
    return x;
  }

private:
  InputDims dims_;
  std::shared_ptr<Tensor> ws_;
  std::unique_ptr<FFT::FFT<5, 3>> fft_;
};
