#pragma once

#include "trajectory.hpp"
#include "functor.hpp"

namespace rl {

struct KSpaceSingle final : Functor<Cx4>
{
  KSpaceSingle(Trajectory const &traj);
  auto operator()(Cx4 const &in) const -> Cx4;

  Re4 weights;
};

} // namespace rl
