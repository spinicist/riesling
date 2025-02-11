#pragma once

#include "basis/basis.hpp"
#include "op/grid.hpp"
#include "op/tensorscale.hpp"
#include "trajectory.hpp"

namespace rl {

struct PreconOpts
{
  std::string type = "single";
  float       λ = 1.e-3f;
};

auto KSpaceSingle(GridOpts<3> const &gridOpts, Trajectory const &traj, float const λ) -> Re2;

auto KSpaceMulti(Cx5 const &smaps, GridOpts<3> const &gridOpts, Trajectory const &traj, float const λ) -> Re3;

auto MakeKSpaceSingle(
  PreconOpts const &opts, GridOpts<3> const &gridOpts, Trajectory const &traj, Index const nC, Index const nS, Index const nT)
  -> TOps::TOp<Cx, 5, 5>::Ptr;

auto MakeKSpaceMulti(
  PreconOpts const &opts, GridOpts<3> const &gridOpts, Trajectory const &traj, Cx5 const &smaps, Index const nS, Index const nT)
  -> TOps::TOp<Cx, 5, 5>::Ptr;

} // namespace rl
