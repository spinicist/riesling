#pragma once

#include "top.hpp"

namespace rl::TOps {

template <typename Scalar_, int Rank, int ImgRank = 3> struct Pad final : TOp<Scalar_, Rank, Rank>
{
  OP_INHERIT(Scalar_, Rank, Rank)

  using ImgDims = Eigen::DSizes<Index, ImgRank>;
  using OtherDims = Eigen::DSizes<Index, Rank - ImgRank>;

  Pad(ImgDims const &imgShape, ImgDims const &padSize, OtherDims const &otherSize);
  Pad(ImgDims const &imgShape, OutDims const oshape);

  OP_DECLARE(Pad)
  void iadjoint(OutCMap const &y, InMap &x) const;
  void iforward(InCMap const &x, OutMap &y) const;

private:
  InDims                                      left_, right_;
  Eigen::array<std::pair<Index, Index>, Rank> paddings_;

  void init();
};
} // namespace rl::TOps
