#pragma once

#include "operator.hpp"

#include "nufft.hpp"
#include "sdc.hpp"
#include "sense.hpp"

namespace rl {

template <typename Op1, typename Op2>
struct Compose final : Operator<typename Op1::Scalar, Op1::InputRank, Op2::OutputRank>
{
  OP_INHERIT(typename Op1::Scalar, Op1::InputRank, Op2::OutputRank)
  Compose(std::shared_ptr<Op1> op1, std::shared_ptr<Op2> op2)
    : Parent(fmt::format("Compose {}+{}", op1->name(), op2->name()), op1->inputDimensions(), op2->outputDimensions())
    , op1_{op1}
    , op2_{op2}
  {
    if (op1_->outputDimensions() != op2_->inputDimensions()) {
      Log::Fail(
        FMT_STRING("{} op1 output: {} did not match op2 input: {}"), this->name(), op1_->outputDimensions(), op2_->inputDimensions());
    }
  }

  auto forward(InputMap x) const -> OutputMap { return op2_->forward(op1_->forward(x)); }
  auto adjoint(OutputMap x) const -> InputMap { return op1_->adjoint(op2_->adjoint(x)); }
  auto adjfwd(InputMap x) const -> InputMap { return op1_->adjoint(op2_->adjfwd(op1_->forward(x))); }
  auto input() const -> InputMap { return op1_->input(); }
  using Parent::adjoint;
  using Parent::forward;

private:
  std::shared_ptr<Op1> op1_;
  std::shared_ptr<Op2> op2_;
};

} // namespace rl
