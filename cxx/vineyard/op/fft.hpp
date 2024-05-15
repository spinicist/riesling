#pragma once

#include "top.hpp"

#include "../fft.hpp"
#include "tensors.hpp"
#include "threads.hpp"

namespace rl::Ops {

template <int Rank, int FFTRank>
struct FFTOp final : TOp<Cx, Rank, Rank>
{
  OP_INHERIT(Cx, Rank, Rank)

  FFTOp(InDims const &dims);
  FFTOp(InMap x);

  using Parent::adjoint;
  using Parent::forward;

  void forward(InCMap const &x, OutMap &y) const;
  void adjoint(OutCMap const &y, InMap &x) const;

private:
  Sz<FFTRank> dims_;
  CxN<FFTRank> ph_;
  Sz<Rank> rsh_, brd_;
};
} // namespace rl::Ops
