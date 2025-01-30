#pragma once

#include "top.hpp"

#include "../basis/basis.hpp"

namespace rl::TOps {

struct f0Segment final : TOp<Cx, 4, 4>
{
  TOP_INHERIT(Cx, 4, 4)
  f0Segment(Re3 const &f0, std::vector<float> const &τ);
  TOP_DECLARE(f0Segment)
  void iforward(InCMap const x, OutMap y) const;
  void iadjoint(OutCMap const y, InMap x) const;

private:
  Re3 f0;
  Cx1 τ;

  Eigen::IndexList<int, FixOne, FixOne, FixOne> v0f123;
  Eigen::IndexList<FixOne, int, int, int>       f0v123;
};

} // namespace rl::TOps
