#pragma once

#include "top.hpp"

#include "../fft.hpp"
#include "../tensors.hpp"
#include "../sys/threads.hpp"

namespace rl::TOps {

template <int Rank, int FFTRank> struct FFT final : TOp<Cx, Rank, Rank>
{
  TOP_INHERIT(Cx, Rank, Rank)

  FFT(InDims const &dims, bool const adjoint = false);
  FFT(InMap x);
  auto inverse() const -> std::shared_ptr<rl::Ops::Op<Cx>> final;

  using Parent::adjoint;
  using Parent::forward;

  void forward(InCMap const &x, OutMap &y) const;
  void adjoint(OutCMap const &y, InMap &x) const;
  void iforward(InCMap const &x, OutMap &y) const;
  void iadjoint(OutCMap const &y, InMap &x) const;

private:
  Sz<FFTRank> dims_;
  bool        adjoint_;
};

} // namespace rl::TOps
