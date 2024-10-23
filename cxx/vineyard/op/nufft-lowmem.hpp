#pragma once

#include "op/top.hpp"

#include "op/grid.hpp"
#include "op/pad.hpp"

namespace rl::TOps {

template <int ND> struct NUFFTLowmem final : TOp<Cx, ND + 1, 3>
{
  TOP_INHERIT(Cx, ND + 1, 3)
  using GType = Grid<ND>;
  NUFFTLowmem(GType::Ptr grid, CxN<ND + 2> const &skern, Sz<ND> const matrix = Sz<ND>());
  TOP_DECLARE(NUFFTLowmem)

  static auto Make(TrajectoryN<ND> const &traj,
                   std::string const     &ktype,
                   float const            osamp,
                   CxN<ND + 2> const     &skern,
                   Basis::CPtr            basis,
                   Sz<ND> const           matrix) -> std::shared_ptr<NUFFTLowmem<ND>>;

  static auto
  Make(TrajectoryN<ND> const &traj, Sz<ND> const &matrix, GridOpts<ND> &opts, CxN<ND + 2> const &skern, Basis::CPtr basis)
    -> std::shared_ptr<NUFFTLowmem<ND>>;

  void iadjoint(OutCMap const &y, InMap &x) const;
  void iforward(InCMap const &x, OutMap &y) const;

private:
  GType::Ptr gridder;
  Cx3 mutable nc1;
  CxN<ND + 2> mutable workspace;
  CxN<ND + 2> skern;
  CxN<ND + 2> mutable smap;
  TOps::Pad<Cx, ND + 2> spad;
  Sz<ND + 2>            sbrd;
  Sz<ND>                fftDims;
  InTensor              apo_;
  InDims                apoBrd_, padLeft_;

  std::array<std::pair<Index, Index>, ND + 1> paddings_;

  void kernToMap(Index const channel) const;
};

} // namespace rl::TOps
