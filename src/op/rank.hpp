#pragma once

#include "operator.hpp"

namespace rl {

template <typename Op>
struct IncreaseOutputRank final : Operator<Op::InputRank, Op::OutputRank + 1>
{
  using Parent = Operator<Op::InputRank, Op::OutputRank + 1>;
  static const size_t InputRank = Parent::InputRank;
  using InputDims = typename Parent::InputDims;
  using Input = typename Parent::Input;
  static const size_t OutputRank = Parent::OutputRank;
  using OutputDims = typename Parent::OutputDims;
  using Output = typename Parent::Output;

  IncreaseOutputRank(Op &op)
    : op_{std::move(op)}
    , y_{outputDimensions()}
  {
  }

  auto inputDimensions() const -> InputDims
  {
    return op_.inputDimensions();
  }

  auto outputDimensions() const -> OutputDims
  {
    return AddBack(op_.outputDimensions(), 1);
  }

  auto forward(Input const &x) const -> Output const &
  {
    this->checkForward(x, "IncreaseRankOp");
    y_ = op_.forward(x).reshape(outputDimensions());
    return y_;
  }

  auto adjoint(Output const &y) const -> Input const &
  {
    this->checkAdjoint(y, "IncreaseRankOp");
    return op_.adjoint(y.reshape(op_.outputDimensions()));
  }

  auto adjfwd(Input const &x) const -> Input 
  {
    return op_.adjfwd(x);
  }

private:
  Op op_;
  mutable Output y_;
};

} // namespace rl