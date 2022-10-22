#pragma once

#include "operator-alloc.hpp"

namespace rl {

struct SenseOp final : OperatorAlloc<Cx, 4, 5>
{
  OPALLOC_INHERIT( Cx, 4, 5 )
  SenseOp(Cx4 const &maps, Index const d0);
  OPALLOC_DECLARE()
  auto nChannels() const -> Index;
  auto mapDimensions() const -> Sz3;

private:
  Cx4 maps_;
  Eigen::IndexList<FixOne, int, int, int, int> resX;
  Eigen::IndexList<int, FixOne, FixOne, FixOne, FixOne> brdX;
  Eigen::IndexList<int, FixOne, int, int, int> resMaps;
  Eigen::IndexList<FixOne, int, FixOne, FixOne, FixOne> brdMaps;
};

} // namespace rl
