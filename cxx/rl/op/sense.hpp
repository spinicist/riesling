#pragma once

#include "top.hpp"

namespace rl::TOps {

struct SENSEOp final : TOp<4, 5>
{
  TOP_INHERIT(4, 5)
  TOP_DECLARE(SENSEOp)
  static auto Make(Cx5 const &maps, Index const nB) -> Ptr;
  SENSEOp(Cx5 const &maps, Index const nB);
  void iforward(InCMap x, OutMap y, float const s = 1.f) const;
  void iadjoint(OutCMap y, InMap x, float const s = 1.f) const;
  auto nChannels() const -> Index;
  auto mapDimensions() const -> Sz3;
  auto maps() const -> Cx5;

private:
  Cx5                                                   maps_;
  Eigen::IndexList<int, int, int, int, FixOne>          resX;
  Eigen::IndexList<FixOne, FixOne, FixOne, FixOne, int> brdX;
  Eigen::IndexList<FixOne, FixOne, FixOne, int, FixOne> brdMaps;
};

auto MakeSENSE(Cx5 const &maps, Index const nB) -> SENSEOp::Ptr;

} // namespace rl::TOps
