#pragma once

#include "operator.hpp"

namespace rl {

template <typename Op>
struct LoopOp final : OperatorAlloc<typename Op::Scalar, Op::InputRank + 1, Op::OutputRank + 1>
{
  OPALLOC_INHERIT( typename Op::Scalar, Op::InputRank + 1, Op::OutputRank + 1 )

  LoopOp(std::shared_ptr<Op> op, Index const N)
    : Parent("LoopOp", AddBack(op->inputDimensions(), N), AddBack(op->outputDimensions(), N))
    , op_{op}
    , N_{N}
  {
  }

  auto forward(InputMap x) const -> OutputMap
  {
    auto const time = this->startForward(x);
    for (Index ii = 0; ii < N_; ii++) {
      Log::Print<Log::Level::Debug>(FMT_STRING("LoopOp Forward Iteration {}"), ii);
      this->output().chip(ii, OutputRank - 1) = op_->forward(ChipMap(x, ii));
    }
    this->finishForward(this->output(), time);
    return this->output();
  }

  auto adjoint(OutputMap y) const -> InputMap
  {
    auto const time = this->startAdjoint(y);
    for (Index ii = 0; ii < N_; ii++) {
      Log::Print<Log::Level::Debug>(FMT_STRING("LoopOp Adjoint Iteration {}"), ii);
      this->input().chip(ii, InputRank - 1) = op_->adjoint(ChipMap(y, ii));
    }
    this->finishAdjoint(this->input(), time);
    return this->input();
  }

  auto adjfwd(InputMap x) const -> InputMap
  {
    for (Index ii = 0; ii < N_; ii++) {
      Log::Print<Log::Level::Debug>(FMT_STRING("LoopOp Adjoint-Forward Iteration {}"), ii);
      this->input().chip(ii, InputRank - 1) = op_->adjfwd(ChipMap(x, ii));
    }
    return this->input();
  }

private:
  std::shared_ptr<Op> op_;
  Index N_;
};

}
