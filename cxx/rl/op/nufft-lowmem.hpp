#pragma once

#include "grid.hpp"
#include "pad.hpp"

namespace rl::TOps {

template <int ND, typename KF = rl::ExpSemi<4>> struct NUFFTLowmem final : TOp<Cx, ND + 1, 3>
{
  TOP_INHERIT(Cx, ND + 1, 3)
  TOP_DECLARE(NUFFTLowmem)
  NUFFTLowmem(GridOpts<ND> const &opts, TrajectoryN<ND> const &traj, CxN<ND + 2> const &skern, Basis::CPtr basis);

  static auto Make(GridOpts<ND> const &opts, TrajectoryN<ND> const &traj, CxN<ND + 2> const &skern, Basis::CPtr basis)
    -> std::shared_ptr<NUFFTLowmem<ND, KF>>;

  void iadjoint(OutCMap y, InMap x) const;
  void iforward(InCMap x, OutMap y) const;

private:
  constexpr static int DC = ND;     // Coils dimension
  constexpr static int DB = ND + 1; // Basis dimension

  Grid<ND, KF>::Ptr gridder;
  Cx3 mutable nc1;
  CxN<ND + 2> mutable workspace;
  CxN<ND + 2> skern;
  CxN<ND + 1> mutable smap;
  TOps::Pad<Cx, ND + 1> spad;
  Sz<ND + 1>            sbrd;
  Sz<ND>                fftDims;
  InTensor              apo_;
  InDims                apoBrd_, padLeft_;

  std::array<std::pair<Index, Index>, ND + 1> paddings_;

  void kernToMap(Index const channel) const;
};

} // namespace rl::TOps
