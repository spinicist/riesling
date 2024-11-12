#pragma once

#include "basis/basis.hpp"
#include "op/grid.hpp"
#include "op/tensorscale.hpp"
#include "sys/args.hpp"
#include "trajectory.hpp"

namespace rl {

struct PreconOpts
{
  std::string type;
  float       bias;
};

auto KSpaceSingle(rl::TOps::Grid<3>::Opts const &gridOpts,
                  Trajectory const              &traj,
                  Basis::CPtr                    basis,
                  float const                    bias,
                  Index const                    nC,
                  Index const                    nS,
                  Index const                    nT) -> typename TOps::TensorScale<Cx, 5, 1, 2>::Ptr;

auto MakeKspacePre(PreconOpts const              &opts,
                   rl::TOps::Grid<3>::Opts const &gridOpts,
                   Trajectory const              &traj,
                   Index const                    nC,
                   Index const                    nS,
                   Index const                    nT,
                   Basis::CPtr                    basis) -> TOps::TOp<Cx, 5, 5>::Ptr;

} // namespace rl
