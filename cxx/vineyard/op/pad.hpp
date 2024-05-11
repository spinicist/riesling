#pragma once

#include "tensorop.hpp"

namespace rl {

template <typename Scalar_, int Rank, int ImgRank = 3>
struct PadOp final : TensorOperator<Scalar_, Rank, Rank>
{
  OP_INHERIT(Scalar_, Rank, Rank)

  using ImgDims = Eigen::DSizes<Index, ImgRank>;
  using OtherDims = Eigen::DSizes<Index, Rank - ImgRank>;

  PadOp(ImgDims const &imgShape, ImgDims const &padSize, OtherDims const &otherSize);
  PadOp(ImgDims const &imgShape, OutDims const oshape);

  OP_DECLARE(PadOp)

private:
  InDims                                      left_, right_;
  Eigen::array<std::pair<Index, Index>, Rank> paddings_;

  void init();
};
} // namespace rl
