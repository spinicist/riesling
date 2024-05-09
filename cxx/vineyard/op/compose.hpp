#pragma once

#include "tensorop.hpp"

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
      Log::Fail("{} op1 output: {} did not match op2 input: {}", this->name, op1_->oshape, op2_->ishape);
    }
  }

  using Parent::adjoint;
  using Parent::forward;

  auto forward(InTensor const &x) const -> OutTensor { return op2_->forward(op1_->forward(x)); }
  auto adjoint(OutTensor const &y) const -> InTensor { return op1_->adjoint(op2_->adjoint(y)); }

  void forward(InCMap const &x, OutMap &y) const
  {
    typename Op1::OutTensor temp(op1_->oshape);
    typename Op1::OutMap    tm(temp.data(), op1_->oshape);
    typename Op1::OutCMap   tcm(temp.data(), op1_->oshape);
    auto const              time = this->startForward(x);
    op1_->forward(x, tm);
    op2_->forward(tcm, y);
    this->finishForward(y, time);
  }

  void adjoint(OutCMap const &y, InMap &x) const
  {
    typename Op1::OutTensor temp(op1_->oshape);
    typename Op1::OutMap    tm(temp.data(), op1_->oshape);
    typename Op1::OutCMap   tcm(temp.data(), op1_->oshape);
    auto const              time = this->startAdjoint(y);
    op2_->adjoint(y, tm);
    op1_->adjoint(tcm, x);
    this->finishAdjoint(x, time);
  }
  // auto adjfwd(InputMap x) const -> InputMap { return op1_->adjoint(op2_->adjfwd(op1_->forward(x))); }

private:
  std::shared_ptr<Op1> op1_;
  std::shared_ptr<Op2> op2_;
};

} // namespace rl
