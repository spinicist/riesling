#pragma once

#include "top.hpp"

namespace rl::TOps {

template <int ND> struct SENSE final : TOp<Cx, ND + 1, ND + 2>
{
  TOP_INHERIT(Cx, ND + 1, ND + 2)
  SENSE(CxN<ND + 2> const &maps, Index const nB);
  TOP_DECLARE(SENSE)
  void iforward(InCMap const x, OutMap y) const;
  void iadjoint(OutCMap const y, InMap x) const;
  auto nChannels() const -> Index;
  auto mapDimensions() const -> Sz<ND>;

private:
  CxN<ND + 2>                                           maps_;
  Eigen::IndexList<int, int, int, FixOne, int>          resX;
  Eigen::IndexList<FixOne, FixOne, FixOne, int, FixOne> brdX;
  Eigen::IndexList<FixOne, FixOne, FixOne, FixOne, int> brdMaps;
};

} // namespace rl::TOps
