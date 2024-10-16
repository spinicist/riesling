#pragma once

#include "op/top.hpp"

#include "op/grid-decant.hpp"
#include "op/pad.hpp"

namespace rl::TOps {

template <int ND> struct NUFFTDecant final : TOp<Cx, ND + 1, 3>
{
  TOP_INHERIT(Cx, ND + 1, 3)
  using GType = GridDecant<ND>;
  NUFFTDecant(GType::Ptr grid, Sz<ND> const matrix = Sz<ND>());
  TOP_DECLARE(NUFFTDecant)

  static auto Make(TrajectoryN<ND> const &traj,
                   std::string const     &ktype,
                   float const            osamp,
                   CxN<ND + 2> const     &skern,
                   Basis::CPtr            basis,
                   Sz<ND> const           matrix,
                   Index const            subgridSz = 8) -> std::shared_ptr<NUFFTDecant<ND>>;

  static auto
  Make(TrajectoryN<ND> const &traj, GridOpts &opts, CxN<ND + 2> const &skern, Basis::CPtr basis, Sz<ND> const matrix)
    -> std::shared_ptr<NUFFTDecant<ND>>;

  void iadjoint(OutCMap const &y, InMap &x) const;
  void iforward(InCMap const &x, OutMap &y) const;

private:
  GType::Ptr gridder;
  InTensor mutable workspace;
  Sz<ND>   fftDims;
  InTensor apo_;
  InDims   apoBrd_, padLeft_;

  std::array<std::pair<Index, Index>, ND + 1> paddings_;
};

} // namespace rl::TOps
