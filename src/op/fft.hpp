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

  Tensor &workspace() const
  {
    return ws_;
  }

  void A() const
  {
    fft_.forward(ws_);
  }

  void Adj() const
  {
    fft_.reverse(ws_);
  }

private:
  InputDims sz_;
  Tensor &ws_;
  FFT::Planned<5, 3> fft_;
};
