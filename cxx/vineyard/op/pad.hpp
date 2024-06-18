#pragma once

#include "top.hpp"

namespace rl::TOps {

template <typename Scalar_, int Rank> struct Pad final : TOp<Scalar_, Rank, Rank>
{
  TOP_INHERIT(Scalar_, Rank, Rank)
  Pad(InDims const ishape, OutDims const oshape);
  auto inverse() const -> std::shared_ptr<rl::Ops::Op<Scalar>> final;
  TOP_DECLARE(Pad)
  void iadjoint(OutCMap const &y, InMap &x) const;
  void iforward(InCMap const &x, OutMap &y) const;

private:
  InDims                                      left_, right_;
  Eigen::array<std::pair<Index, Index>, Rank> paddings_;

  void init();
};

template <typename Scalar_, int Rank> struct Crop final : TOp<Scalar_, Rank, Rank>
{
  TOP_INHERIT(Scalar_, Rank, Rank)
  Crop(InDims const big, OutDims const small);
  auto inverse() const -> std::shared_ptr<rl::Ops::Op<Scalar>> final;
  TOP_DECLARE(Crop)
  void iadjoint(OutCMap const &y, InMap &x) const;
  void iforward(InCMap const &x, OutMap &y) const;

private:
  InDims                                      left_, right_;
  Eigen::array<std::pair<Index, Index>, Rank> paddings_;

  void init();
};
} // namespace rl::TOps
