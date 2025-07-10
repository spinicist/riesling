#pragma once

#include "top-impl.hpp"

#include "../log/log.hpp"

namespace rl::TOps {

template <typename Op, int Rank> struct ReshapeInput final : TOp<Rank, Op::OutRank>
{
  TOP_INHERIT(Rank, Op::OutRank)
  using Parent::adjoint;
  using Parent::forward;
  using Ptr = std::shared_ptr<ReshapeInput>;

  ReshapeInput(std::shared_ptr<Op> op, Sz<Rank> const ish)
    : Parent("ReshapeInput", ish, op->oshape)
    , op_{op}
  {
    if (Product(ish) != Product(op->ishape)) {
      throw Log::Failure("TOp", "ReshapeInput shape {} does not match {}", ish, op->ishape);
    }
  }

  void forward(InCMap x, OutMap y, float const s = 1.f) const
  {
    auto const          time = this->startForward(x, y, false);
    typename Op::InCMap xm(x.data(), op_->ishape);
    typename Op::OutMap ym(y.data(), op_->oshape);
    op_->forward(xm, ym, s);
    this->finishForward(y, time, false);
  }

  void adjoint(OutCMap y, InMap x, float const s = 1.f) const
  {
    auto const           time = this->startAdjoint(y, x, false);
    typename Op::OutCMap ym(y.data(), op_->oshape);
    typename Op::InMap   xm(x.data(), op_->ishape);
    op_->adjoint(ym, xm, s);
    this->finishAdjoint(x, time, false);
  }

  void iforward(InCMap x, OutMap y, float const s = 1.f) const
  {
    auto const          time = this->startForward(x, y, true);
    typename Op::InCMap xm(x.data(), op_->ishape);
    typename Op::OutMap ym(y.data(), op_->oshape);
    op_->iforward(xm, ym, s);
    this->finishForward(y, time, true);
  }

  void iadjoint(OutCMap y, InMap x, float const s = 1.f) const
  {
    auto const           time = this->startAdjoint(y, x, true);
    typename Op::OutCMap ym(y.data(), op_->oshape);
    typename Op::InMap   xm(x.data(), op_->ishape);
    op_->iadjoint(ym, xm, s);
    this->finishAdjoint(x, time, true);
  }

private:
  std::shared_ptr<Op> op_;
};

template <typename Op, int Rank> auto MakeReshapeInput(std::shared_ptr<Op> op, Sz<Rank> const ish)
  -> ReshapeInput<Op, Rank>::Ptr
{
  return std::make_shared<ReshapeInput<Op, Rank>>(op, ish);
}

template <typename Op, int Rank> struct ReshapeOutput final : TOp<Op::InRank, Rank>
{
  TOP_INHERIT(Op::InRank, Rank)
  using Parent::adjoint;
  using Parent::forward;
  using Ptr = std::shared_ptr<ReshapeOutput>;

  ReshapeOutput(std::shared_ptr<Op> op, Sz<Rank> const osh)
    : Parent("ReshapeOutput", op->ishape, osh)
    , op_{op}
  {
    if (Product(osh) != Product(op->oshape)) {
      throw Log::Failure("TOp", "ReshapeInput shape {} does not match {}", osh, op->oshape);
    }
  }

  void forward(InCMap x, OutMap y, float const s) const
  {
    auto const          time = this->startForward(x, y, false);
    typename Op::InCMap xm(x.data(), op_->ishape);
    typename Op::OutMap ym(y.data(), op_->oshape);
    op_->forward(xm, ym, s);
    this->finishForward(y, time, false);
  }

  void adjoint(OutCMap y, InMap x, float const s) const
  {
    auto const           time = this->startAdjoint(y, x, false);
    typename Op::OutCMap ym(y.data(), op_->oshape);
    typename Op::InMap   xm(x.data(), op_->ishape);
    op_->adjoint(ym, xm, s);
    this->finishAdjoint(x, time, false);
  }

  void iforward(InCMap x, OutMap y, float const s = 1.f) const
  {
    auto const          time = this->startForward(x, y, true);
    typename Op::InCMap xm(x.data(), op_->ishape);
    typename Op::OutMap ym(y.data(), op_->oshape);
    op_->iforward(xm, ym, s);
    this->finishForward(y, time, true);
  }

  void iadjoint(OutCMap y, InMap x, float const s = 1.f) const
  {
    auto const           time = this->startAdjoint(y, x, true);
    typename Op::OutCMap ym(y.data(), op_->oshape);
    typename Op::InMap   xm(x.data(), op_->ishape);
    op_->iadjoint(ym, xm, s);
    this->finishAdjoint(x, time, true);
  }

private:
  std::shared_ptr<Op> op_;
};

template <typename Op, int Rank> auto MakeReshapeOutput(std::shared_ptr<Op> op, Sz<Rank> const osh)
  -> ReshapeOutput<Op, Rank>::Ptr
{
  return std::make_shared<ReshapeOutput<Op, Rank>>(op, osh);
}

} // namespace rl::TOps
