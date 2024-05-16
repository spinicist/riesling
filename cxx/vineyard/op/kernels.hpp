#pragma once

#include "top.hpp"

namespace rl {

template <typename Sc, int ND, int NK> struct Kernels final : TOp<Sc, ND, ND + 1>
{
  static_assert(NK < ND);
  OP_INHERIT(Sc, ND, ND + 1)
  using Parent::adjoint;
  using Parent::forward;

  Kernels(InDims const ish, Sz<NK> const kDims, Sz<NK> const kW);
  void forward(InCMap const &x, OutMap &y) const;
  void adjoint(OutCMap const &y, InMap &x) const;

private:
  Sz<NK> kDims_, kW_;
  Sz<ND> kSz_;
};

} // namespace rl