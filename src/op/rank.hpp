#pragma once

#include "operator.hpp"

namespace rl {

template <typename Op>
struct IncreaseOutputRank final : Operator<typename Op::Scalar, Op::InputRank, Op::OutputRank + 1>
{
  OP_INHERIT(typename Op::Scalar, Op::InputRank, Op::OutputRank + 1)

  IncreaseOutputRank(std::unique_ptr<Op> &&op)
    : Parent("IncreaseOutputRankOp", op->inputDimensions(), AddBack(op->outputDimensions(), 1))
    , op_{std::move(op)}
  {
  }

  auto forward(InputMap x) const -> OutputMap
  {
    auto const time = this->startForward(x);
    auto y = op_->forward(x);
    OutputMap y2(y.data(), this->outputDimensions());
    this->finishForward(y2, time);
    return y2;
  }

  auto adjoint(OutputMap y) const -> InputMap
  {
    auto const time = this->startAdjoint(y);
    typename Op::OutputMap y2(y.data(), op_->outputDimensions());
    auto x = op_->adjoint(y2);
    this->finishAdjoint(x, time);
    return x;
  }

  auto adjfwd(InputMap x) const -> InputMap { return op_->adjfwd(x); }

private:
  std::unique_ptr<Op> op_;
};

} // namespace rl