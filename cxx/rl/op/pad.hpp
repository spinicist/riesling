#pragma once

#include "top.hpp"

namespace rl::TOps {

template <typename Scalar_, int Rank> struct Pad final : TOp<Scalar_, Rank, Rank>
{
  TOP_INHERIT(Scalar_, Rank, Rank)
  Pad(InDims const ishape, OutDims const oshape);
  TOP_DECLARE(Pad)
  void iadjoint(OutCMap y, InMap x, float const s = 1.f) const;
  void iforward(InCMap x, OutMap y, float const s = 1.f) const;

private:
  InDims                                      left_, right_;
  Eigen::array<std::pair<Index, Index>, Rank> paddings_;

  void init();
};

} // namespace rl::TOps
