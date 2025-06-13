#pragma once

#include "grid-decant.hpp"
#include "pad.hpp"

namespace rl::TOps {

template <int ND, typename KF = rl::ExpSemi<4>> struct NUFFTDecant final : TOp<Cx, ND + 1, 3>
{
  TOP_INHERIT(Cx, ND + 1, 3)
  TOP_DECLARE(NUFFTDecant)

  NUFFTDecant(GridOpts<ND> const &opts, TrajectoryN<ND> const &traj, CxN<ND + 2> const &skern, Basis::CPtr basis);

  static auto Make(GridOpts<ND> const &opts, TrajectoryN<ND> const &traj, CxN<ND + 2> const &skern, Basis::CPtr basis)
    -> std::shared_ptr<NUFFTDecant<ND>>;

  void iadjoint(OutCMap y, InMap x, float const s) const;
  void iforward(InCMap x, OutMap y, float const s) const;

private:
  GridDecant<ND, KF> gridder;
  InTensor mutable workspace;
  Sz<ND>   fftDims;
  InTensor apo_;
  InDims   apoBrd_, padLeft_;

  std::array<std::pair<Index, Index>, ND + 1> paddings_;
};

} // namespace rl::TOps
