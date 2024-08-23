#pragma once

#include "top.hpp"

#include "basis/basis.hpp"

namespace rl::TOps {

struct SENSE final : TOp<Cx, 4, 5>
{
  TOP_INHERIT(Cx, 4, 5)
  SENSE(Cx5 const &maps, Index const nB = 1);
  TOP_DECLARE(SENSE)
  void iforward(InCMap const &x, OutMap &y) const;
  void iadjoint(OutCMap const &y, InMap &x) const;
  auto nChannels() const -> Index;
  auto mapDimensions() const -> Sz3;

private:
  Cx5                                                   maps_;
  Eigen::IndexList<int, FixOne, int, int, int>          resX;
  Eigen::IndexList<FixOne, int, FixOne, FixOne, FixOne> brdX;
  Eigen::IndexList<int, FixOne, FixOne, FixOne, FixOne> brdMaps;
};

struct EstimateKernels final : TOp<Cx, 5, 5>
{
  TOP_INHERIT(Cx, 5, 5)
  EstimateKernels(Cx4 const &img, Index const nC);
  TOP_DECLARE(SENSE)
  void iforward(InCMap const &x, OutMap &y) const;
  void iadjoint(OutCMap const &y, InMap &x) const;
  auto nChannels() const -> Index;
  auto mapDimensions() const -> Sz3;

private:
  Cx4                                                   img_;
  Eigen::IndexList<int, FixOne, int, int, int>          res_;
  Eigen::IndexList<FixOne, int, FixOne, FixOne, FixOne> brd_;
};

struct VCCSENSE final : TOp<Cx, 4, 6>
{
  TOP_INHERIT(Cx, 4, 6)
  VCCSENSE(Cx5 const &maps, Index const nB = 1);
  TOP_DECLARE(VCCSENSE)
  void iforward(InCMap const &x, OutMap &y) const;
  void iadjoint(OutCMap const &y, InMap &x) const;
  auto nChannels() const -> Index;
  auto mapDimensions() const -> Sz3;

private:
  Cx5                                                   maps_;
  Eigen::IndexList<int, FixOne, int, int, int>          resX;
  Eigen::IndexList<FixOne, int, FixOne, FixOne, FixOne> brdX;
  Eigen::IndexList<int, FixOne, FixOne, FixOne, FixOne> brdMaps;
};

} // namespace rl::TOps
