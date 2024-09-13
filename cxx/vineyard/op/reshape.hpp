#pragma once

#include "op/top.hpp"

#include "log.hpp"

namespace rl::TOps {

template <typename Op, int Rank> struct ReshapeInput final : TOp<typename Op::Scalar, Rank, Op::OutRank>
{
  TOP_INHERIT(typename Op::Scalar, Rank, Op::OutRank)
  using Parent::adjoint;
  using Parent::forward;

  ReshapeInput(std::shared_ptr<Op> op, Sz<Rank> const ish)
    : Parent("ReshapeInput", ish, op->oshape)
    , op_{op}
  {
    if (Product(ish) != Product(op->ishape)) { throw Log::Failure("TOp", "ReshapeInput shape {} does not match {}", ish, op->ishape); }
  }

  void forward(InCMap const &x, OutMap &y) const
  {
    auto const                time = this->startForward(x, y, false);
    typename Op::InCMap const xm(x.data(), op_->ishape);
    typename Op::OutMap       ym(y.data(), op_->oshape);
    op_->forward(xm, ym);
    this->finishForward(y, time, false);
  }

  void adjoint(OutCMap const &y, InMap &x) const
  {
    auto const                 time = this->startAdjoint(y, x, false);
    typename Op::OutCMap const ym(y.data(), op_->oshape);
    typename Op::InMap         xm(x.data(), op_->ishape);
    op_->adjoint(ym, xm);
    this->finishAdjoint(x, time, false);
  }

  void iforward(InCMap const &x, OutMap &y) const
  {
    auto const                time = this->startForward(x, y, true);
    typename Op::InCMap const xm(x.data(), op_->ishape);
    typename Op::OutMap       ym(y.data(), op_->oshape);
    op_->iforward(xm, ym);
    this->finishForward(y, time, true);
  }

  void iadjoint(OutCMap const &y, InMap &x) const
  {
    auto const                 time = this->startAdjoint(y, x, true);
    typename Op::OutCMap const ym(y.data(), op_->oshape);
    typename Op::InMap         xm(x.data(), op_->ishape);
    op_->iadjoint(ym, xm);
    this->finishAdjoint(x, time, true);
  }

private:
  std::shared_ptr<Op> op_;
};

template <typename Op, int Rank> struct ReshapeOutput final : TOp<typename Op::Scalar, Op::InRank, Rank>
{
  TOP_INHERIT(typename Op::Scalar, Op::InRank, Rank)
  using Parent::adjoint;
  using Parent::forward;

  ReshapeOutput(std::shared_ptr<Op> op, Sz<Rank> const osh)
    : Parent("ReshapeOutput", op->ishape, osh)
    , op_{op}
  {
    if (Product(osh) != Product(op->oshape)) { throw Log::Failure("TOp", "ReshapeInput shape {} does not match {}", osh, op->oshape); }
  }

  void forward(InCMap const &x, OutMap &y) const
  {
    auto const                time = this->startForward(x, y, false);
    typename Op::InCMap const xm(x.data(), op_->ishape);
    typename Op::OutMap       ym(y.data(), op_->oshape);
    op_->forward(xm, ym);
    this->finishForward(y, time, false);
  }

  void adjoint(OutCMap const &y, InMap &x) const
  {
    auto const                 time = this->startAdjoint(y, x, false);
    typename Op::OutCMap const ym(y.data(), op_->oshape);
    typename Op::InMap         xm(x.data(), op_->ishape);
    op_->adjoint(ym, xm);
    this->finishAdjoint(x, time, false);
  }

  void iforward(InCMap const &x, OutMap &y) const
  {
    auto const                time = this->startForward(x, y, true);
    typename Op::InCMap const xm(x.data(), op_->ishape);
    typename Op::OutMap       ym(y.data(), op_->oshape);
    op_->iforward(xm, ym);
    this->finishForward(y, time, true);
  }

  void iadjoint(OutCMap const &y, InMap &x) const
  {
    auto const                 time = this->startAdjoint(y, x, true);
    typename Op::OutCMap const ym(y.data(), op_->oshape);
    typename Op::InMap         xm(x.data(), op_->ishape);
    op_->iadjoint(ym, xm);
    this->finishAdjoint(x, time, true);
  }

private:
  std::shared_ptr<Op> op_;
};

} // namespace rl::TOps
