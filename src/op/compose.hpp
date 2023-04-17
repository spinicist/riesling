#pragma once

#include "tensorop.hpp"

#include "nufft.hpp"
#include "sdc.hpp"
#include "sense.hpp"

namespace rl {

template <typename Op1, typename Op2>
struct Compose final : TensorOperator<typename Op1::Scalar, Op1::InRank, Op2::OutRank>
{
  OP_INHERIT(typename Op1::Scalar, Op1::InRank, Op2::OutRank)
  Compose(std::shared_ptr<Op1> op1, std::shared_ptr<Op2> op2)
    : Parent(fmt::format("Compose {}+{}", op1->name, op2->name), op1->ishape, op2->oshape)
    , op1_{op1}
    , op2_{op2}
  {
    if (op1_->oshape != op2_->ishape) {
      Log::Fail(
        FMT_STRING("{} op1 output: {} did not match op2 input: {}"), this->name, op1_->oshape, op2_->ishape);
    }
  }

  auto forward(InTensor const &x) const -> OutTensor { return op2_->forward(op1_->forward(x)); }
  auto adjoint(OutTensor const &y) const -> InTensor { return op1_->adjoint(op2_->adjoint(y)); }
  // auto adjfwd(InputMap x) const -> InputMap { return op1_->adjoint(op2_->adjfwd(op1_->forward(x))); }

  using Parent::adjoint;
  using Parent::forward;

private:
  std::shared_ptr<Op1> op1_;
  std::shared_ptr<Op2> op2_;
};

} // namespace rl
