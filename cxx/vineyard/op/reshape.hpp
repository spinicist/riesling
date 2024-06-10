#pragma once

#include "op/top.hpp"

namespace rl::TOps {

template <typename Op, int Rank> struct ReshapeInput final : TOp<typename Op::Scalar, Rank, Op::OutRank>
{
  OP_INHERIT(typename Op::Scalar, Rank, Op::OutRank)
  using Parent::adjoint;
  using Parent::forward;

  ReshapeInput(std::shared_ptr<Op> op, Sz<Rank> const ish)
    : Parent("ReshapeInput", ish, op->oshape)
    , op_{op}
  {
    if (Product(ish) != Product(op->ishape)) { Log::Fail("ReshapeInput shape {} does not match {}", ish, op->ishape); }
  }

  void forward(InCMap const &x, OutMap &y) const
  {
    auto const                time = this->startForward(x, y);
    typename Op::InCMap const xm(x.data(), op_->ishape);
    typename Op::OutMap       ym(y.data(), op_->oshape);
    op_->forward(xm, ym);
    this->finishForward(y, time);
  }

  void adjoint(OutCMap const &y, InMap &x) const
  {
    auto const                 time = this->startAdjoint(y, x);
    typename Op::OutCMap const ym(y.data(), op_->oshape);
    typename Op::InMap         xm(x.data(), op_->ishape);
    op_->adjoint(ym, xm);
    this->finishAdjoint(x, time);
  }

private:
  std::shared_ptr<Op> op_;
};

} // namespace rl::TOps