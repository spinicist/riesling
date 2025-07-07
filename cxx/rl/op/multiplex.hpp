#pragma once

#include "top.hpp"

namespace rl::TOps {

template <int ND> struct Multiplex final : TOp<ND, ND + 1>
{
  TOP_INHERIT(ND, ND + 1)
  using Parent::adjoint;
  using Parent::forward;

  Multiplex(InDims const ish, Index const nSlab)
    : Parent("MultiplexOp", ish, AddBack(FirstN<InRank - 1>(ish), ish[InRank - 1] / nSlab, nSlab))
  {
  }

  void forward(InCMap x, OutMap y) const
  {
    auto const  time = this->startForward(x, y, false);
    Index const nSlab = oshape[InRank];
    Sz<InRank>  st;
    Sz<InRank>  sz = ishape;
    sz[InRank - 1] /= nSlab;
    for (Index is = 0; is < nSlab; is++) {
      y.template chip<InRank>(is) = x.slice(st, sz);
      st[InRank - 1] += sz[InRank - 1];
    }
    this->finishForward(y, time, false);
  }

  void adjoint(OutCMap y, InMap x) const
  {
    auto const  time = this->startAdjoint(y, x, false);
    Index const nSlab = oshape[InRank];
    Sz<InRank>  st;
    Sz<InRank>  sz = ishape;
    sz[InRank - 1] /= nSlab;
    for (Index is = 0; is < nSlab; is++) {
      x.slice(st, sz) = y.template chip<InRank>(is);
      st[InRank - 1] += sz[InRank - 1];
    }
    this->finishAdjoint(x, time, false);
  }
};

} // namespace rl::TOps