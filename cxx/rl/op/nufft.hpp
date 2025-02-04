#pragma once

#include "../op/grid.hpp"
#include "../op/pad.hpp"
#include "../op/top.hpp"

namespace rl::TOps {

template <int ND, typename KF = rl::ExpSemi<4>> struct NUFFT final : TOp<Cx, ND + 2, 3>
{
  TOP_INHERIT(Cx, ND + 2, 3)
  NUFFT(GridOpts<ND> const &opts, TrajectoryN<ND> const &traj, Index const nC, Basis::CPtr basis);
  TOP_DECLARE(NUFFT)

  static auto Make(GridOpts<ND> const &opts, TrajectoryN<ND> const &traj, Index const nC, Basis::CPtr basis)
    -> std::shared_ptr<NUFFT<ND, KF>>;

  void iadjoint(OutCMap const y, InMap x) const;
  void iforward(InCMap const x, OutMap y) const;

private:
  Grid<ND, KF>::Ptr gridder;
  InTensor mutable workspace;
  Sz<ND>   fftDims;
  InTensor apo_;
  InDims   apoBrd_, padLeft_;

  std::array<std::pair<Index, Index>, ND + 2> paddings_;
};

// Utility function to build a complete NUFFT pipeline over all slabs and timepoints
auto NUFFTAll(
  GridOpts<3> const &gridOpts, Trajectory const &traj, Index const nC, Index const nS, Index const nT, Basis::CPtr basis)
  -> TOps::TOp<Cx, 6, 5>::Ptr;

} // namespace rl::TOps
