#pragma once

#include "func/functor.hpp"
#include "trajectory.hpp"

namespace rl {

auto KSpaceSingle(Trajectory const &traj) -> Re2;

std::unique_ptr<Functor<Cx4>> make_pre(std::string const &type, Trajectory const &traj);

} // namespace rl
