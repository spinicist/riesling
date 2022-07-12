#pragma once

#include "../tensorOps.h"
#include "operator.hpp"

namespace rl {

struct SenseOp final : Operator<4, 5>
{
  SenseOp(Cx4 const &maps, Index const d0)
    : maps_{std::move(maps)}
    , inSz_{d0, maps_.dimension(1), maps_.dimension(2), maps_.dimension(3)}
    , outSz_{maps_.dimension(0), d0, maps_.dimension(1), maps_.dimension(2), maps_.dimension(3)}
  {
    resX.set(1, d0);
    resX.set(2, maps_.dimension(1));
    resX.set(3, maps_.dimension(2));
    resX.set(4, maps_.dimension(3));
    brdX.set(0, maps_.dimension(0));

    resMaps.set(0, maps_.dimension(0));
    resMaps.set(2, maps_.dimension(1));
    resMaps.set(3, maps_.dimension(2));
    resMaps.set(4, maps_.dimension(3));
    brdMaps.set(1, d0);
  }

  InputDims inputDimensions() const
  {
    return inSz_;
  }
  OutputDims outputDimensions() const
  {
    return outSz_;
  }

  template <typename T>
  auto A(T const &x) const
  {
    return (x.reshape(resX).broadcast(brdX) * maps_.reshape(resMaps).broadcast(brdMaps));
  }

  template <typename T>
  auto Adj(T const &x) const
  {
    return ConjugateSum(x, maps_.reshape(resMaps).broadcast(brdMaps));
  }

private:
  Cx4 maps_;
  Sz4 inSz_;
  Sz5 outSz_;
  Eigen::IndexList<FixOne, int, int, int, int> resX;
  Eigen::IndexList<int, FixOne, FixOne, FixOne, FixOne> brdX;
  Eigen::IndexList<int, FixOne, int, int, int> resMaps;
  Eigen::IndexList<FixOne, int, FixOne, FixOne, FixOne> brdMaps;
};
} // namespace rl
