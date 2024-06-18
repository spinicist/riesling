#pragma once

#include "ops.hpp"

namespace rl::TOps {

template <typename Op> struct Loop final : TOp<typename Op::Scalar, Op::InRank + 1, Op::OutRank + 1>
{
  TOP_INHERIT(typename Op::Scalar, Op::InRank + 1, Op::OutRank + 1)
  using Parent::adjoint;
  using Parent::forward;

  Loop(std::shared_ptr<Op> op, Index const N)
    : Parent("Loop", AddBack(op->ishape, N), AddBack(op->oshape, N))
    , op_{op}
    , N_{N}
  {
  }

  void forward(InCMap const &x, OutMap &y) const
  {
    assert(x.dimensions() == this->ishape);
    assert(y.dimensions() == this->oshape);
    auto const time = this->startForward(x, y, false);
    for (Index ii = 0; ii < N_; ii++) {
      typename Op::InCMap xchip(x.data() + Product(op_->ishape) * ii, op_->ishape);
      typename Op::OutMap ychip(y.data() + Product(op_->oshape) * ii, op_->oshape);
      op_->forward(xchip, ychip);
    }
    this->finishForward(y, time, false);
  }

  void adjoint(OutCMap const &y, InMap &x) const
  {
    assert(x.dimensions() == this->ishape);
    assert(y.dimensions() == this->oshape);
    auto const time = this->startAdjoint(y, x, false);
    for (Index ii = 0; ii < N_; ii++) {
      typename Op::OutCMap ychip(y.data() + Product(op_->oshape) * ii, op_->oshape);
      typename Op::InMap   xchip(x.data() + Product(op_->ishape) * ii, op_->ishape);
      Log::Debug("Loop op {}/{}", ii, N_);
      op_->adjoint(ychip, xchip);
    }
    this->finishAdjoint(x, time, false);
  }

  void iforward(InCMap const &x, OutMap &y) const
  {
    assert(x.dimensions() == this->ishape);
    assert(y.dimensions() == this->oshape);
    auto const time = this->startForward(x, y, true);
    for (Index ii = 0; ii < N_; ii++) {
      typename Op::InCMap xchip(x.data() + Product(op_->ishape) * ii, op_->ishape);
      typename Op::OutMap ychip(y.data() + Product(op_->oshape) * ii, op_->oshape);
      op_->iforward(xchip, ychip);
    }
    this->finishForward(y, time, true);
  }

  void iadjoint(OutCMap const &y, InMap &x) const
  {
    assert(x.dimensions() == this->ishape);
    assert(y.dimensions() == this->oshape);
    auto const time = this->startAdjoint(y, x, true);
    for (Index ii = 0; ii < N_; ii++) {
      typename Op::OutCMap ychip(y.data() + Product(op_->oshape) * ii, op_->oshape);
      typename Op::InMap   xchip(x.data() + Product(op_->ishape) * ii, op_->ishape);
      Log::Debug("Loop op {}/{}", ii, N_);
      op_->iadjoint(ychip, xchip);
    }
    this->finishAdjoint(x, time, true);
  }

private:
  std::shared_ptr<Op> op_;
  Index               N_;
};

} // namespace rl::TOps
