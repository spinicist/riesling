#pragma once

#include "top.hpp"

#include "basis/basis.hpp"

namespace rl::TOps {

struct SENSE final : TOp<Cx, 4, 5>
{
  TOP_INHERIT(Cx, 4, 5)
  SENSE(Cx5 const &maps, bool const vcc, Index const nB);
  TOP_DECLARE(SENSE)
  void iforward(InCMap const &x, OutMap &y) const;
  void iadjoint(OutCMap const &y, InMap &x) const;
  auto nChannels() const -> Index;
  auto mapDimensions() const -> Sz3;

private:
  bool                                                  vcc;
  Cx5                                                   maps_;
  Eigen::IndexList<int, FixOne, int, int, int>          resX;
  Eigen::IndexList<FixOne, int, FixOne, FixOne, FixOne> brdX;
  Eigen::IndexList<int, FixOne, FixOne, FixOne, FixOne> brdMaps;
};

} // namespace rl::TOps
