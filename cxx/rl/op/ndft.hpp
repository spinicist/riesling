#pragma once

#include "../basis/basis.hpp"
#include "../op/top.hpp"

namespace rl::TOps {

template <int NDim> struct NDFT final : TOp<NDim + 2, 3>
{
  TOP_INHERIT(NDim + 2, 3)
  NDFT(Sz<NDim> const matrix, Re3 const &traj, Index const nC, Basis::CPtr basis);
  TOP_DECLARE(NDFT)

  void iadjoint(OutCMap y, InMap x) const;
  void iforward(InCMap x, OutMap y) const;

  static auto Make(Sz<NDim> const matrix, Re3 const &traj, Index const nC, Basis::CPtr basis) -> Ptr;
  void        addOffResonance(Eigen::Tensor<float, NDim> const &f0map, float const t0, float const tSamp);
  auto        M(float const λ, Index const nS, Index const nT) const -> TOps::TOp<5, 5>::Ptr; // Left (k-space) Pre-conditioner diag(AA')

private:
  Re3         traj;
  Re2         xc;
  Re1         Δf, t;
  Index       N, nSamp, nTrace;
  float       scale;
  Basis::CPtr basis;
};

} // namespace rl::TOps
