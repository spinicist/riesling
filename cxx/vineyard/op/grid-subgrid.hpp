#pragma once

#include "types.hpp"

namespace rl {

template <int ND, bool VCC> struct Subgrid
{
  Sz<ND>               gridSize, minCorner, maxCorner;
  std::vector<int32_t> indices;

  auto empty() const -> bool;
  auto count() const -> Index;
  auto size() const -> Sz<ND>;

  template<bool isVCC>
  void gridToSubgrid(CxNCMap<ND + 2 + VCC> const &x, CxN<ND + 2> &sx) const;
};

} // namespace rl
