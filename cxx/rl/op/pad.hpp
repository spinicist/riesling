#pragma once

#include "top.hpp"

namespace rl::TOps {

template <int Rank> struct Pad final : TOp<Rank, Rank>
{
  TOP_INHERIT(Rank, Rank)
  Pad(InDims const ishape, OutDims const oshape);
  TOP_DECLARE(Pad)
  static auto Make(InDims const ishape, OutDims const oshape) -> Ptr;

  void iadjoint(OutCMap y, InMap x, float const s = 1.f) const;
  void iforward(InCMap x, OutMap y, float const s = 1.f) const;

private:
  InDims                                      left_, right_;
  Eigen::array<std::pair<Index, Index>, Rank> paddings_;

  void init();
};

} // namespace rl::TOps
