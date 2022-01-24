#pragma once

#include "../fft_plan.h"
#include "operator.h"

template <int Rank, int FFTRank = 3>
struct FFTOp final : Operator<Rank, Rank>
{
  using Parent = Operator<Rank, Rank>;
  using Input = typename Parent::Input;
  using InputDims = typename Parent::InputDims;
  using Output = typename Parent::Output;
  using OutputDims = typename Parent::OutputDims;
  using Tensor = typename Eigen::Tensor<Cx, Rank>;

  FFTOp(Tensor &ws)
    : ws_{ws}
    , fft_{ws_}
  {
  }

  InputDims inputDimensions() const
  {
    return ws_.dimensions();
  }

  OutputDims outputDimensions() const
  {
    return ws_.dimensions();
  }

  template <typename T>
  Tensor const &A(T const &x) const
  {
    ws_.device(Threads::GlobalDevice()) = x;
    fft_.forward(ws_);
    return ws_;
  }

  template <typename T>
  Tensor const &Adj(T const &x) const
  {
    ws_.device(Threads::GlobalDevice()) = x;
    fft_.reverse(ws_);
    return ws_;
  }

private:
  InputDims sz_;
  Tensor &ws_;
  FFT::Planned<5, 3> fft_;
};
