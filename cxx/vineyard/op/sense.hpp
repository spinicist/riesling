#pragma once

#include "tensorop.hpp"

namespace rl {

struct SenseOp final : TensorOperator<Cx, 4, 5>
{
  OP_INHERIT(Cx, 4, 5)
  SenseOp(Cx5 const &maps, Index const frames);
  OP_DECLARE(SenseOp)
  auto nChannels() const -> Index;
  auto mapDimensions() const -> Sz3;

private:
  Cx5                                                   maps_;
  Eigen::IndexList<FixOne, int, int, int, int>          resX;
  Eigen::IndexList<int, FixOne, FixOne, FixOne, FixOne> brdX;
  Eigen::IndexList<FixOne, int, FixOne, FixOne, FixOne> brdMaps;
};

} // namespace rl
