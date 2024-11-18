#pragma once

#include "basis/basis.hpp"
#include "op/grid.hpp"
#include "op/tensorscale.hpp"
#include "sys/args.hpp"
#include "trajectory.hpp"

namespace rl {

struct PreconOpts
{
  std::string type = "single";
  float       λ = 1.e-3f;
};

auto KSpaceSingle(rl::TOps::Grid<3>::Opts const &gridOpts, Trajectory const &traj, Basis::CPtr basis, float const λ) -> Re2;

auto KSpaceMulti(
  Cx5 const &smaps, rl::TOps::Grid<3>::Opts const &gridOpts, Trajectory const &traj, Basis::CPtr basis, float const λ) -> Re3;

auto MakeKSpaceSingle(PreconOpts const              &opts,
                      rl::TOps::Grid<3>::Opts const &gridOpts,
                      Trajectory const              &traj,
                      Index const                    nC,
                      Index const                    nS,
                      Index const                    nT,
                      Basis::CPtr                    basis) -> TOps::TOp<Cx, 5, 5>::Ptr;

} // namespace rl
