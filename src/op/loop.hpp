#pragma once

#include "operator.hpp"

namespace rl {

template <typename Op>
struct LoopOp final : TensorOperator<typename Op::Scalar, Op::InRank + 1, Op::OutRank + 1>
{
  OP_INHERIT( typename Op::Scalar, Op::InRank + 1, Op::OutRank + 1 )

  LoopOp(std::shared_ptr<Op> op, Index const N)
    : Parent("LoopOp", AddBack(op->ishape, N), AddBack(op->oshape, N))
    , op_{op}
    , N_{N}
  {
  }

  void forward(InCMap const &x, OutMap &y) const
  {
    auto const time = this->startForward(x);
    for (Index ii = 0; ii < N_; ii++) {
      Log::Print<Log::Level::Debug>(FMT_STRING("LoopOp Forward Iteration {}"), ii);
      y.template chip<OutRank - 1>(ii) = op_->forward(x.template chip<InRank - 1>(ii));
    }
    this->finishForward(y, time);
  }

  void adjoint(OutCMap const &y, InMap &x) const
  {
    auto const time = this->startAdjoint(y);
    for (Index ii = 0; ii < N_; ii++) {
      Log::Print<Log::Level::Debug>(FMT_STRING("LoopOp Adjoint Iteration {}"), ii);
      x.template chip<InRank - 1>(ii) = op_->adjoint(y.template chip<InRank - 1>(ii));
    }
    this->finishAdjoint(x, time);
  }

  // auto adjfwd(InputMap x) const -> InputMap
  // {
  //   for (Index ii = 0; ii < N_; ii++) {
  //     Log::Print<Log::Level::Debug>(FMT_STRING("LoopOp Adjoint-Forward Iteration {}"), ii);
  //     this->input().chip(ii, InRank - 1) = op_->adjfwd(ChipMap(x, ii));
  //   }
  //   return this->input();
  // }

private:
  std::shared_ptr<Op> op_;
  Index N_;
};

}
