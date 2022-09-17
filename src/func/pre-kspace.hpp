#pragma once

#include "trajectory.hpp"
#include "functor.hpp"

namespace rl {

struct KSpaceSingle final : Functor<Cx3>
{
  KSpaceSingle(Trajectory const &traj);
  auto operator()(Cx3 const &in) const -> Cx3;

  Re3 weights;
};

} // namespace rl
