#pragma once

#include "op/top.hpp"

#include "op/grid.hpp"
#include "op/pad.hpp"

namespace rl::TOps {

template <int NDim, bool VCC = false> struct NUFFT final : TOp<Cx, NDim + 2 + VCC, 3>
{
  TOP_INHERIT(Cx, NDim + 2 + VCC, 3)
  NUFFT(TrajectoryN<NDim> const &traj,
        std::string const       &ktype,
        float const              osamp,
        Index const              nC,
        Basis::CPtr              basis,
        Sz<NDim> const           matrix = Sz<NDim>(),
        Index const              subgridSz = 32,
        Index const              nBatches = 1);
  TOP_DECLARE(NUFFT)

  static auto
  Make(TrajectoryN<NDim> const &traj, GridOpts &opts, Index const nC, Basis::CPtr basis, Sz<NDim> const matrix = Sz<NDim>())
    -> std::shared_ptr<NUFFT<NDim, VCC>>;

  void iadjoint(OutCMap const &y, InMap &x) const;
  void iforward(InCMap const &x, OutMap &y) const;

private:
  Grid<NDim, VCC> gridder;
  InTensor mutable workspace;

  Index const batches;
  InDims      batchShape_;
  Sz<NDim>    fftDims;
  CxN<NDim>   fftPh;

  InTensor apo_;
  InDims   apoBrd_, padLeft_;

  std::array<std::pair<Index, Index>, NDim + 2 + VCC> paddings_;
};

// Utility function to build a complete NUFFT pipeline over all slabs and timepoints
auto NUFFTAll(GridOpts         &gridOpts,
              Trajectory const &traj,
              Index const       nC,
              Index const       nS,
              Index const       nT,
              Basis::CPtr       basis,
              Sz3 const         shape) -> TOps::TOp<Cx, 6, 5>::Ptr;

} // namespace rl::TOps
