#pragma once

#include "basis/basis.hpp"
#include "op/grid-opts.hpp"
#include "op/top.hpp"
#include "precon-opts.hpp"
#include "trajectory.hpp"

namespace rl {

template <int ND>
auto KSpaceSingle(GridOpts<ND> const &gridOpts, TrajectoryN<ND> const &traj, float const max = 1.f, Basis::CPtr basis = nullptr)
  -> Re2;
auto KSpaceMulti(Cx5 const         &smaps,
                 GridOpts<3> const &gridOpts,
                 Trajectory const  &traj,
                 float const        max = 1.f,
                 Basis::CPtr        basis = nullptr) -> Re3;

template <int ND, int NB> auto MakeKSpacePrecon(PreconOpts const      &opts,
                                                GridOpts<ND> const    &gridOpts,
                                                TrajectoryN<ND> const &traj,
                                                Index const            nC,
                                                Sz<NB> const           bshape) -> TOps::TOp<3 + NB>::Ptr;

template <int ND, int NB> auto MakeKSpacePrecon(PreconOpts const      &opts,
                                                GridOpts<ND> const    &gridOpts,
                                                TrajectoryN<ND> const &traj,
                                                Cx5 const             &smaps,
                                                Sz<NB> const           bshape) -> TOps::TOp<3 + NB, 3 + NB>::Ptr;

} // namespace rl
