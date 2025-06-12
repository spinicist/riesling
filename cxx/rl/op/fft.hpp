#pragma once

#include "top.hpp"

#include "../fft.hpp"
#include "../tensors.hpp"
#include "../sys/threads.hpp"

namespace rl::TOps {

template <int Rank, int FFTRank> struct FFT final : TOp<Cx, Rank, Rank>
{
  TOP_INHERIT(Cx, Rank, Rank)

  FFT(InDims const &shape, bool const adjoint = false);
  FFT(InDims const &shape, Sz<FFTRank> const dims, bool const adjoint = false);
  FFT(InMap x);

  using Parent::adjoint;
  using Parent::forward;

  void forward(InCMap x, OutMap y) const;
  void adjoint(OutCMap y, InMap x) const;
  void iforward(InCMap x, OutMap y) const;
  void iadjoint(OutCMap y, InMap x) const;

private:
  Sz<FFTRank> dims_;
  bool        adjoint_;
};

} // namespace rl::TOps
