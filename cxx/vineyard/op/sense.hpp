#pragma once

#include "top.hpp"

namespace rl::TOps {

struct SENSE final : TOp<Cx, 4, 5>
{
  OP_INHERIT(Cx, 4, 5)
  SENSE(Cx5 const &maps, Index const frames);
  OP_DECLARE(SENSE)
  auto nChannels() const -> Index;
  auto mapDimensions() const -> Sz3;

private:
  Cx5                                                   maps_;
  Eigen::IndexList<FixOne, int, int, int, int>          resX;
  Eigen::IndexList<int, FixOne, FixOne, FixOne, FixOne> brdX;
  Eigen::IndexList<FixOne, int, FixOne, FixOne, FixOne> brdMaps;
};

} // namespace rl::TOps
