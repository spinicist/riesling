#pragma once

#include "top.hpp"

#include "../log/log.hpp"

namespace rl::TOps {

template <int ID, int OD, typename Op> struct Loop final : TOp<typename Op::Scalar, Op::InRank + 1, Op::OutRank + 1>
{
  TOP_INHERIT(typename Op::Scalar, Op::InRank + 1, Op::OutRank + 1)
  using Parent::adjoint;
  using Parent::forward;
  using Ptr = std::shared_ptr<Loop>;

  Loop(typename Op::Ptr op, Index const N)
    : Parent("Loop",
             Concatenate(FirstN<ID>(op->ishape), Sz1{N}, LastN<InRank - 1 - ID>(op->ishape)),
             Concatenate(FirstN<OD>(op->oshape), Sz1{N}, LastN<OutRank - 1 - OD>(op->oshape)))
    , op_{op}
    , N_{N}
  {
  }

  void forward(InCMap const x, OutMap y) const
  {
    assert(x.dimensions() == this->ishape);
    assert(y.dimensions() == this->oshape);
    auto const time = this->startForward(x, y, false);
    for (Index ii = 0; ii < N_; ii++) {
      Log::Debug("Op", "Loop {}/{}", ii, N_);
      y.template chip<OD>(ii) = op_->forward(x.template chip<ID>(ii));
    }
    this->finishForward(y, time, false);
  }

  void adjoint(OutCMap const y, InMap x) const
  {
    assert(x.dimensions() == this->ishape);
    assert(y.dimensions() == this->oshape);
    auto const time = this->startAdjoint(y, x, false);
    for (Index ii = 0; ii < N_; ii++) {
      Log::Debug("Op", "Loop {}/{}", ii, N_);
      x.template chip<ID>(ii) = op_->adjoint(y.template chip<OD>(ii));
    }
    this->finishAdjoint(x, time, false);
  }

  void iforward(InCMap const x, OutMap y) const
  {
    assert(x.dimensions() == this->ishape);
    assert(y.dimensions() == this->oshape);
    auto const time = this->startForward(x, y, true);
    for (Index ii = 0; ii < N_; ii++) {
      Log::Debug("Op", "Loop {}/{}", ii, N_);
      y.template chip<OD>(ii) += op_->forward(x.template chip<ID>(ii));
    }
    this->finishForward(y, time, true);
  }

  void iadjoint(OutCMap const y, InMap x) const
  {
    assert(x.dimensions() == this->ishape);
    assert(y.dimensions() == this->oshape);
    auto const time = this->startAdjoint(y, x, true);
    for (Index ii = 0; ii < N_; ii++) {
      Log::Debug("Op", "Loop {}/{}", ii, N_);
      x.template chip<ID>(ii) += op_->adjoint(y.template chip<OD>(ii));
    }
    this->finishAdjoint(x, time, true);
  }

private:
  std::shared_ptr<Op> op_;
  Index               N_;
};

template <int ID, int OD, typename Op> auto MakeLoop(std::shared_ptr<Op> op, Index const N) -> Loop<ID, OD, Op>::Ptr
{
  return std::make_shared<Loop<ID, OD, Op>>(op, N);
}

} // namespace rl::TOps
