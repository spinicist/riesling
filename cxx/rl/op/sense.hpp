#pragma once

#include "top.hpp"

namespace rl::TOps {

template <int ND> struct SENSEOp final : TOp<Cx, ND + 1, ND + 2>
{
  TOP_INHERIT(Cx, ND + 1, ND + 2)
  SENSEOp(CxN<ND + 2> const &maps, Index const nB);
  SENSEOp(CxN<ND + 2> const &kern, Sz<ND> const mat, float const os, Index const nB);
  TOP_DECLARE(SENSEOp)
  void iforward(InCMap const x, OutMap y) const;
  void iadjoint(OutCMap const y, InMap x) const;
  auto nChannels() const -> Index;
  auto mapDimensions() const -> Sz<ND>;
  auto maps() const -> CxN<ND + 2>;

private:
  CxN<ND + 2>                                           maps_;
  Eigen::IndexList<int, int, int, FixOne, int>          resX;
  Eigen::IndexList<FixOne, FixOne, FixOne, int, FixOne> brdX;
  Eigen::IndexList<FixOne, FixOne, FixOne, FixOne, int> brdMaps;
};

template <int ND> auto MakeSENSE(CxN<ND + 2> const &maps, Index const nB) -> SENSEOp<ND>::Ptr;
template <int ND> auto MakeSENSE(CxN<ND + 2> const &kern, Sz<ND> const mat, float const os, Index const nB) -> SENSEOp<ND>::Ptr;

} // namespace rl::TOps
