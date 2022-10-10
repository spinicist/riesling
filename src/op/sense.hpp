#pragma once

#include "operator.hpp"

namespace rl {

struct SenseOp final : Operator<4, 5>
{
  SenseOp(Cx4 const &maps, Index const d0);

  InputDims inputDimensions() const;
  OutputDims outputDimensions() const;

  auto forward(Input const &x) const -> Output const &;
  auto adjoint(Output const &y) const -> Input const &;

private:
  Cx4 maps_;
  Sz4 inSz_;
  Sz5 outSz_;
  Eigen::IndexList<FixOne, int, int, int, int> resX;
  Eigen::IndexList<int, FixOne, FixOne, FixOne, FixOne> brdX;
  Eigen::IndexList<int, FixOne, int, int, int> resMaps;
  Eigen::IndexList<FixOne, int, FixOne, FixOne, FixOne> brdMaps;
  mutable Input x_;
  mutable Output y_;
};

} // namespace rl
