#pragma once

#include "log.hpp"
#include "operator-alloc.hpp"
#include "tensorOps.hpp"

namespace rl {

template <typename Scalar_, int Rank, int ImgRank = 3>
struct PadOp final : OperatorAlloc<Scalar_, Rank, Rank>
{
  OPALLOC_INHERIT(Scalar_, Rank, Rank)

  using ImgDims = Eigen::DSizes<Index, ImgRank>;
  using OtherDims = Eigen::DSizes<Index, Rank - ImgRank>;

  // Note how init works to get the input dimensions set up before allocating storage
  PadOp(ImgDims const &imgSize, ImgDims const &padSize, OtherDims const &otherSize = {});
  PadOp(OutputMap yStorage, ImgDims const &imgSize, ImgDims const &padSize, OtherDims const &otherSize = {});
  PadOp(InputMap x, OutputMap y);

  OPALLOC_DECLARE()

private:
  InputDims left_, right_;
  Eigen::array<std::pair<Index, Index>, Rank> paddings_;
  float scale_;

  void init(ImgDims const &imgSize, ImgDims const &padSize);
};
} // namespace rl
