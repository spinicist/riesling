#pragma once

#include "basis/basis.hpp"
#include "op/grid-opts.hpp"
#include "op/tensorscale.hpp"
#include "trajectory.hpp"

namespace rl {

struct PreconOpts
{
  std::string type = "single";
  float       λ = 1.e-3f;
};

template <int ND>
auto KSpaceSingle(GridOpts<ND> const &gridOpts, TrajectoryN<ND> const &traj, float const λ, Basis::CPtr basis = nullptr) -> Re2;
auto KSpaceMulti(
  Cx5 const &smaps, GridOpts<3> const &gridOpts, Trajectory const &traj, float const λ, Basis::CPtr basis = nullptr) -> Re3;

template <int ND> auto MakeKSpacePrecon(PreconOpts const      &opts,
                                        GridOpts<ND> const    &gridOpts,
                                        TrajectoryN<ND> const &traj,
                                        Index const            nC,
                                        Index const            nS,
                                        Index const            nT) -> TOps::TOp<Cx, 5, 5>::Ptr;

auto MakeKSpacePrecon(
  PreconOpts const &opts, GridOpts<3> const &gridOpts, Trajectory const &traj, Cx5 const &smaps, Index const nS, Index const nT)
  -> TOps::TOp<Cx, 5, 5>::Ptr;

} // namespace rl
