#pragma once

#include "top.hpp"

namespace rl::TOps {

struct SENSEOp final : TOp<Cx, 4, 5>
{
  TOP_INHERIT(Cx, 4, 5)
  SENSEOp(Cx5 const &maps, Index const nB);
  TOP_DECLARE(SENSEOp)
  void iforward(InCMap x, OutMap y, float const s) const;
  void iadjoint(OutCMap y, InMap x, float const s) const;
  auto nChannels() const -> Index;
  auto mapDimensions() const -> Sz3;
  auto maps() const -> Cx5;

private:
  Cx5                                                   maps_;
  Eigen::IndexList<int, int, int, FixOne, int>          resX;
  Eigen::IndexList<FixOne, FixOne, FixOne, int, FixOne> brdX;
  Eigen::IndexList<FixOne, FixOne, FixOne, FixOne, int> brdMaps;
};

auto MakeSENSE(Cx5 const &maps, Index const nB) -> SENSEOp::Ptr;

} // namespace rl::TOps
