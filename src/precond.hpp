#pragma once

#include "func/functor.hpp"
#include "trajectory.hpp"

namespace rl {

auto KSpaceSingle(Trajectory const &traj, std::optional<Re2> const basis = std::nullopt) -> Re2;

std::shared_ptr<Functor<Cx4>> make_pre(std::string const &type, Trajectory const &traj);

} // namespace rl
