#pragma once

#include "top.hpp"

namespace rl::TOps {

template <typename Sc, int ND, int NK> struct Hankel final : TOp<Sc, ND, ND + 2>
{
  static_assert(NK < ND);
  OP_INHERIT(Sc, ND, ND + 2)
  using Parent::adjoint;
  using Parent::forward;

  Hankel(InDims const ish, Sz<NK> const kDims, Sz<NK> const kW, const bool sphere = false, const bool virt = false);
  void forward(InCMap const &x, OutMap &y) const;
  void adjoint(OutCMap const &y, InMap &x) const;

private:
  Sz<NK> kDims_, kW_;
  Sz<ND> kSz_;
  bool sphere_, virt_;
};

} // namespace rl