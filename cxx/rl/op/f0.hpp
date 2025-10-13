#pragma once

#include "top.hpp"

#include "../basis/basis.hpp"

namespace rl::TOps {

struct f0Segment final : TOp<4, 4>
{
  TOP_INHERIT(4, 4)
  f0Segment(Re3 const &f0, float const τ0, float const τacq, Index const Nτ, Index const Nacq);
  TOP_DECLARE(f0Segment)
  void iforward(InCMap x, OutMap y, float const s = 1.f) const;
  void iadjoint(OutCMap y, InMap x, float const s = 1.f) const;

  auto basis() const -> Basis::CPtr;

private:
  Re3 f0;
  Cx1 τ;
  Basis b;

  Eigen::IndexList<int, int, int, FixOne> v012f3;
  Eigen::IndexList<FixOne, FixOne, FixOne, int>       f012v3;
};

} // namespace rl::TOps
