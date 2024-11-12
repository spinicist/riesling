#pragma once

#include "op/top.hpp"

#include "op/grid.hpp"
#include "op/pad.hpp"

namespace rl::TOps {

template <int ND> struct NUFFTLowmem final : TOp<Cx, ND + 1, 3>
{
  TOP_INHERIT(Cx, ND + 1, 3)
  using GType = Grid<ND>;
  NUFFTLowmem(Grid<ND>::Opts const &opts, TrajectoryN<ND> const &traj, CxN<ND + 2> const &skern, Basis::CPtr basis);
  TOP_DECLARE(NUFFTLowmem)

  static auto Make(Grid<ND>::Opts const &opts, TrajectoryN<ND> const &traj, CxN<ND + 2> const &skern, Basis::CPtr basis)
    -> std::shared_ptr<NUFFTLowmem<ND>>;

  void iadjoint(OutCMap const &y, InMap &x) const;
  void iforward(InCMap const &x, OutMap &y) const;

private:
  GType::Ptr gridder;
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
