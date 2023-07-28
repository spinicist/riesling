#pragma once

#include "ops.hpp"

namespace rl {

template <typename Op>
struct LoopOp final : TensorOperator<typename Op::Scalar, Op::InRank + 1, Op::OutRank + 1>
{
  OP_INHERIT(typename Op::Scalar, Op::InRank + 1, Op::OutRank + 1)
  using Parent::adjoint;
  using Parent::forward;

  LoopOp(std::shared_ptr<Op> op, Index const N)
    : Parent("LoopOp", AddBack(op->ishape, N), AddBack(op->oshape, N))
    , op_{op}
    , N_{N}
  {
  }

  void forward(InCMap const &x, OutMap &y) const
  {
    assert(x.dimensions() == this->ishape);
    assert(y.dimensions() == this->oshape);
    auto const time = this->startForward(x);
    for (Index ii = 0; ii < N_; ii++) {
      typename Op::InCMap xchip(x.data() + Product(op_->ishape) * ii, op_->ishape);
      typename Op::OutMap ychip(y.data() + Product(op_->oshape) * ii, op_->oshape);
      op_->forward(xchip, ychip);
    }
    this->finishForward(y, time);
  }

  void adjoint(OutCMap const &y, InMap &x) const
  {
    assert(x.dimensions() == this->ishape);
    assert(y.dimensions() == this->oshape);
    auto const time = this->startAdjoint(y);
    for (Index ii = 0; ii < N_; ii++) {
      typename Op::OutCMap ychip(y.data() + Product(op_->oshape) * ii, op_->oshape);
      typename Op::InMap   xchip(x.data() + Product(op_->ishape) * ii, op_->ishape);
      Log::Print<Log::Level::Debug>("Loop op {}/{}", ii, N_);
      op_->adjoint(ychip, xchip);
    }
    this->finishAdjoint(x, time);
  }

  // auto adjfwd(InputMap x) const -> InputMap
  // {
  //   for (Index ii = 0; ii < N_; ii++) {
  //     Log::Print<Log::Level::Debug>("LoopOp Adjoint-Forward Iteration {}", ii);
  //     this->input().chip(ii, InRank - 1) = op_->adjfwd(ChipMap(x, ii));
  //   }
  //   return this->input();
  // }

private:
  std::shared_ptr<Op> op_;
  Index               N_;
};

} // namespace rl
