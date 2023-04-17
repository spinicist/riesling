#pragma once

#include "operator.hpp"

namespace rl {

template <typename Op>
struct IncreaseOutputRank final : TensorOperator<typename Op::Scalar, Op::InRank, Op::OutRank + 1>
{
  OP_INHERIT(typename Op::Scalar, Op::InRank, Op::OutRank + 1)

  IncreaseOutputRank(std::shared_ptr<Op> op)
    : Parent("IncreaseOutputRankOp", op->ishape, AddBack(op->oshape, 1))
    , op_{op}
  {
  }

  void forward(InCMap const &x, OutMap &y) const
  {
    auto const time = this->startForward(x);
    typename Op::OutMap y2(y.data(), op_->oshape);
    y2 = op_->forward(x);
    this->finishForward(y, time);
  }

  void adjoint(OutCMap const &y, InMap &x) const
  {
    auto const time = this->startAdjoint(y);
    typename Op::OutCMap y2(y.data(), op_->oshape);
    x = op_->adjoint(y2);
    this->finishAdjoint(x, time);
  }

  // auto adjfwd(InputMap x) const -> InputMap { return op_->adjfwd(x); }

private:
  std::shared_ptr<Op> op_;
};

} // namespace rl