#pragma once

#include "top.hpp"

#include "../fft.hpp"
#include "../tensors.hpp"
#include "../sys/threads.hpp"

namespace rl::TOps {

template <int Rank, int FFTRank> struct FFT final : TOp<Rank, Rank>
{
  TOP_INHERIT(Rank, Rank)

  FFT(InDims const &shape, bool const adjoint = false);
  FFT(InDims const &shape, Sz<FFTRank> const dims, bool const adjoint = false);
  FFT(InMap x);

  using Parent::adjoint;
  using Parent::forward;
  using Ptr = std::shared_ptr<FFT<Rank, FFTRank>>;
  static auto Make(InDims const &shape, bool const adjoint = false) -> Ptr;

  void forward(InCMap x, OutMap y, float const s) const;
  void adjoint(OutCMap y, InMap x, float const s) const;
  void iforward(InCMap x, OutMap y, float const s) const;
  void iadjoint(OutCMap y, InMap x, float const s) const;

private:
  Sz<FFTRank> dims_;
  bool        adjoint_;
};

} // namespace rl::TOps
