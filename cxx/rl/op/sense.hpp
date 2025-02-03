#pragma once

#include "top.hpp"

#include "../basis/basis.hpp"

namespace rl::TOps {

struct SENSE final : TOp<Cx, 4, 5>
{
  TOP_INHERIT(Cx, 4, 5)
  SENSE(Cx5 const &maps, Index const nB);
  TOP_DECLARE(SENSE)
  void iforward(InCMap const &x, OutMap &y) const;
  void iadjoint(OutCMap const &y, InMap &x) const;
  auto nChannels() const -> Index;
  auto mapDimensions() const -> Sz3;

private:
  Cx5                                                   maps_;
  Eigen::IndexList<int, int, int, FixOne, int>          resX;
  Eigen::IndexList<FixOne, FixOne, FixOne, int, FixOne> brdX;
  Eigen::IndexList<FixOne, FixOne, FixOne, FixOne, int> brdMaps;
};

} // namespace rl::TOps
