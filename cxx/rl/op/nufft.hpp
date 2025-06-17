#pragma once

#include "apodize.hpp"
#include "grid.hpp"
#include "pad.hpp"

namespace rl::TOps {

template <int ND, typename KF = ExpSemi<4>> struct NUFFT final : TOp<Cx, ND + 2, 3>
{
  TOP_INHERIT(Cx, ND + 2, 3)
  NUFFT(GridOpts<ND> const &opts, TrajectoryN<ND> const &traj, Index const nC, Basis::CPtr basis);
  TOP_DECLARE(NUFFT)

  static auto Make(GridOpts<ND> const &opts, TrajectoryN<ND> const &traj, Index const nC, Basis::CPtr basis)
    -> TOp<Cx, ND + 2, 3>::Ptr;

  void iadjoint(OutCMap y, InMap x, float const s = 1.f) const;
  void iforward(InCMap x, OutMap y, float const s = 1.f) const;

private:
  Grid<ND, KF>       gridder;
  Apodize<ND, 2, KF> apo;
  InTensor mutable workspace;
  Sz<ND> fftDims;
};

} // namespace rl::TOps
