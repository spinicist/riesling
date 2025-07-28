#pragma once

#include "apodize.hpp"
#include "grid.hpp"
#include "pad.hpp"

namespace rl::TOps {

template <int ND, typename KF = rl::ExpSemi<4>> struct NUFFTLowmem final : TOp<ND + 1, 3>
{
  TOP_INHERIT(ND + 1, 3)
  TOP_DECLARE(NUFFTLowmem)
  NUFFTLowmem(GridOpts<ND> const &opts, TrajectoryN<ND> const &traj, CxN<ND + 2> const &skern, Basis::CPtr basis);

  static auto Make(GridOpts<ND> const &opts, TrajectoryN<ND> const &traj, CxN<ND + 2> const &skern, Basis::CPtr basis)
    -> std::shared_ptr<NUFFTLowmem<ND, KF>>;

  void iadjoint(OutCMap y, InMap x, float const s) const;
  void iforward(InCMap x, OutMap y, float const s) const;

private:
  constexpr static int DC = ND + 1; // Coils dimension
  constexpr static int DB = ND;     // Basis dimension

  Grid<ND, KF>       gridder;
  Apodize<ND, 1, KF> apo;
  Cx3 mutable nc1;
  CxN<ND + 2> mutable workspace;
  CxN<ND + 2> skern;
  CxN<ND + 1> mutable smap;
  TOps::Pad<ND + 1> spad;
  Sz<ND + 1>        sbrd;
  Sz<ND>            fftDims;
  void              kernToMap(Index const channel) const;
};

} // namespace rl::TOps
