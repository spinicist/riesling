#pragma once

#include "top.hpp"

namespace rl::TOps {

/*
 * This represents Op2 * Op1
 */
template <typename Op1, typename Op2> struct Compose final : TOp<typename Op1::Scalar, Op1::InRank, Op2::OutRank>
{
  TOP_INHERIT(typename Op1::Scalar, Op1::InRank, Op2::OutRank)
  Compose(std::shared_ptr<Op1> op1, std::shared_ptr<Op2> op2)
    : Parent(fmt::format("{}+{}", op1->name, op2->name), op1->ishape, op2->oshape)
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
    auto const              time = this->startForward(x, y, false);
    op1_->forward(x, tm);
    op2_->forward(tcm, y);
    this->finishForward(y, time, false);
  }

  void adjoint(OutCMap const &y, InMap &x) const
  {
    typename Op1::OutTensor temp(op1_->oshape);
    typename Op1::OutMap    tm(temp.data(), op1_->oshape);
    typename Op1::OutCMap   tcm(temp.data(), op1_->oshape);
    auto const              time = this->startAdjoint(y, x, false);
    op2_->adjoint(y, tm);
    op1_->adjoint(tcm, x);
    this->finishAdjoint(x, time, false);
  }

  void iforward(InCMap const &x, OutMap &y) const
  {
    typename Op1::OutTensor temp(op1_->oshape);
    typename Op1::OutMap    tm(temp.data(), op1_->oshape);
    typename Op1::OutCMap   tcm(temp.data(), op1_->oshape);
    auto const              time = this->startForward(x, y, true);
    op1_->forward(x, tm);
    op2_->iforward(tcm, y);
    this->finishForward(y, time, true);
  }

  void iadjoint(OutCMap const &y, InMap &x) const
  {
    typename Op1::OutTensor temp(op1_->oshape);
    typename Op1::OutMap    tm(temp.data(), op1_->oshape);
    typename Op1::OutCMap   tcm(temp.data(), op1_->oshape);
    auto const              time = this->startAdjoint(y, x, true);
    op2_->adjoint(y, tm);
    op1_->iadjoint(tcm, x);
    this->finishAdjoint(x, time, true);
  }

private:
  std::shared_ptr<Op1> op1_;
  std::shared_ptr<Op2> op2_;
};

} // namespace rl::TOps
