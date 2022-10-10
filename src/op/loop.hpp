#pragma once

#include "operator.hpp"

namespace rl {

template <typename Op>
struct LoopOp final : Operator<Op::InputRank + 1, Op::OutputRank + 1>
{
  using Parent = Operator<Op::InputRank + 1, Op::OutputRank + 1>;
  static const size_t InputRank = Parent::InputRank;
  using InputDims = typename Parent::InputDims;
  using Input = typename Parent::Input;
  static const size_t OutputRank = Parent::OutputRank;
  using OutputDims = typename Parent::OutputDims;
  using Output = typename Parent::Output;

  LoopOp(Op &op, Index const N)
    : op_{std::move(op)}
    , N_{N}
    , x_{inputDimensions()}
    , y_{outputDimensions()}
  {
  }

  auto inputDimensions() const -> InputDims
  {
    return AddBack(op_.inputDimensions(), N_);
  }

  auto outputDimensions() const -> OutputDims
  {
    return AddBack(op_.outputDimensions(), N_);
  }

  auto forward(Input const &x) const -> Output const &
  {
    for (Index ii = 0; ii < N_; ii++) {
      Log::Print<Log::Level::Debug>(FMT_STRING("LoopOp Forward Iteration {}"), ii);
      y_.chip(ii, OutputRank - 1) = op_.forward(x.chip(ii, InputRank - 1));
    }
    return y_;
  }

  auto adjoint(Output const &y) const -> Input const &
  {
    for (Index ii = 0; ii < N_; ii++) {
      Log::Print<Log::Level::Debug>(FMT_STRING("LoopOp Adjoint Iteration {}"), ii);
      x_.chip(ii, InputRank - 1) = op_.adjoint(y.chip(ii, OutputRank - 1));
    }
    return x_;
  }

  auto adjfwd(Input const &x) const -> Input
  {
    for (Index ii = 0; ii < N_; ii++) {
      Log::Print<Log::Level::Debug>(FMT_STRING("LoopOp Adjoint-Forward Iteration {}"), ii);
      x_.chip(ii, InputRank - 1) = op_.adjfwd(x.chip(ii, InputRank - 1));
    }
    return x_;
  }

private:
  Op op_;
  Index N_;
  mutable Input x_;
  mutable Output y_;
};

}
