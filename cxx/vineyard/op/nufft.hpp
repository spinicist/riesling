#pragma once

#include "op/top.hpp"

#include "op/grid.hpp"
#include "op/pad.hpp"

namespace rl::TOps {

template <int ND, bool VCC = false> struct NUFFT final : TOp<Cx, ND + 2 + VCC, 3>
{
  TOP_INHERIT(Cx, ND + 2 + VCC, 3)
  using GType = Grid<ND, VCC>;
  NUFFT(GType::Ptr grid, Sz<ND> const matrix = Sz<ND>(), Index const subgridSz = 8);
  TOP_DECLARE(NUFFT)

  static auto Make(TrajectoryN<ND> const &traj,
                   std::string const     &ktype,
                   float const            osamp,
                   Index const            nC,
                   Basis::CPtr            basis,
                   Sz<ND> const           matrix,
                   Index const            subgridSz = 8) -> std::shared_ptr<NUFFT<ND, VCC>>;

  static auto Make(TrajectoryN<ND> const &traj, GridOpts<ND> const &opts, Index const nC, Basis::CPtr basis)
    -> std::shared_ptr<NUFFT<ND, VCC>>;

  void iadjoint(OutCMap const &y, InMap &x) const;
  void iforward(InCMap const &x, OutMap &y) const;

private:
  GType::Ptr gridder;
  InTensor mutable workspace;
  Sz<ND>   fftDims;
  InTensor apo_;
  InDims   apoBrd_, padLeft_;

  std::array<std::pair<Index, Index>, ND + 2 + VCC> paddings_;
};

// Utility function to build a complete NUFFT pipeline over all slabs and timepoints
auto NUFFTAll(
  GridOpts<3> const &gridOpts, Trajectory const &traj, Index const nC, Index const nS, Index const nT, Basis::CPtr basis)
  -> TOps::TOp<Cx, 6, 5>::Ptr;

} // namespace rl::TOps
