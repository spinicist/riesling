#pragma once

#include "types.hpp"

namespace rl {

template <int ND> struct Subgrid
{
  Sz<ND>               minCorner, maxCorner;
  auto size() const -> Sz<ND>;

  template<bool hasVCC, bool isVCC>
  void gridToSubgrid(CxNCMap<ND + 2 + hasVCC> const &x, CxN<ND + 2> &sx) const;
  template<bool hasVCC, bool isVCC>
  void subgridToGrid(CxNCMap<ND + 2> const &sx, CxNMap<ND + 2 + hasVCC> &x) const;
};

} // namespace rl
