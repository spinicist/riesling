#pragma once

#include "op/top.hpp"

#include "op/grid.hpp"
#include "op/pad.hpp"

namespace rl::TOps {

template <int NDim, bool VCC = false> struct NUFFT final : TOp<Cx, NDim + 2 + VCC, 3>
{
  OP_INHERIT(Cx, NDim + 2 + VCC, 3)
  NUFFT(Sz<NDim> const           matrix,
        TrajectoryN<NDim> const &traj,
        std::string const       &ktype,
        float const              osamp,
        Index const              nC,
        Basis<Cx> const         &basis = IdBasis<Scalar>(),
        Index const              subgridSz = 32,
        Index const              splitSz = 16384,
        Index const              nBatches = 1);
  OP_DECLARE(NUFFT)

  static auto Make(Sz<NDim> const matrix, TrajectoryN<NDim> const &traj, GridOpts &opts, Index const nC, Basis<Cx> const &basis)
    -> std::shared_ptr<NUFFT<NDim, VCC>>;

  void iadjoint(OutCMap const &y, InMap &x) const;
  void iforward(InCMap const &x, OutMap &y) const;

private:
  Grid<Cx, NDim, VCC> gridder;
  InTensor mutable workspace;

  Index const batches;
  InDims      batchShape_;
  Sz<NDim>    fftDims;
  CxN<NDim>   fftPh;

  InTensor apo_;
  InDims   apoBrd_, padLeft_;

  std::array<std::pair<Index, Index>, NDim + 2 + VCC> paddings_;
};

} // namespace rl::TOps
