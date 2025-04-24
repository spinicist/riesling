#pragma once

#include "top.hpp"

#include "../basis/basis.hpp"

namespace rl::TOps {

struct f0Segment final : TOp<Cx, 4, 4>
{
  TOP_INHERIT(Cx, 4, 4)
  f0Segment(Re3 const &f0, float const τacq, Index const Nτ, Index const Nacq);
  TOP_DECLARE(f0Segment)
  void iforward(InCMap const x, OutMap y) const;
  void iadjoint(OutCMap const y, InMap x) const;

  auto basis() const -> Basis::CPtr;

private:
  Re3 f0;
  Cx1 τ;
  Basis b;

  Eigen::IndexList<int, int, int, FixOne> v012f3;
  Eigen::IndexList<FixOne, FixOne, FixOne, int>       f012v3;
};

} // namespace rl::TOps
