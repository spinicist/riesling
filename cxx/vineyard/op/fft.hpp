#pragma once

#include "top.hpp"

#include "../fft.hpp"
#include "tensors.hpp"
#include "threads.hpp"

namespace rl::TOps {

template <int Rank, int FFTRank> struct FFT final : TOp<Cx, Rank, Rank>
{
  OP_INHERIT(Cx, Rank, Rank)

  FFT(InDims const &dims);
  FFT(InMap x);

  using Parent::adjoint;
  using Parent::forward;

  void forward(InCMap const &x, OutMap &y) const;
  void adjoint(OutCMap const &y, InMap &x) const;

private:
  Sz<FFTRank>  dims_;
  CxN<FFTRank> ph_;
};

} // namespace rl::TOps
