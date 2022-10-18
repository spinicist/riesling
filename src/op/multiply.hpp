#pragma once

#include "operator.hpp"

#include "nufft.hpp"
#include "sdc.hpp"
#include "sense.hpp"

namespace rl {

template <typename Op1, typename Op2>
struct MultiplyOp final : Operator<typename Op1::Scalar, Op1::InputRank, Op2::OutputRank>
{
  OP_INHERIT(typename Op1::Scalar, Op1::InputRank, Op2::OutputRank)
  MultiplyOp(std::string const &name, std::unique_ptr<Op1> &op1, std::unique_ptr<Op2> &op2)
    : Parent(name, op1->inputDimensions(), op2->outputDimensions())
    , op1_{std::move(op1)}
    , op2_{std::move(op2)}
  {
    if (op1_->outputDimensions() != op2_->inputDimensions()) {
      Log::Fail(
        FMT_STRING("{} op1 output: {} did not match op2 input: {}"), name, op1_->outputDimensions(), op2_->inputDimensions());
    }
  }
  using Parent::inputDimensions;
  using Parent::outputDimensions;
  auto forward(InputMap x) const -> OutputMap { return op2_->forward(op1_->forward(x)); }
  auto adjoint(OutputMap x) const -> InputMap { return op1_->adjoint(op2_->adjoint(x)); }
  auto adjfwd(InputMap x) const -> InputMap { return op1_->adjoint(op2_->adjfwd(op1_->forward(x))); }
  using Parent::adjoint;
  using Parent::forward;

private:
  std::unique_ptr<Op1> op1_;
  std::unique_ptr<Op2> op2_;
};

} // namespace rl
