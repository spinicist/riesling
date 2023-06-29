#pragma once

#include "op/ops.hpp"
#include "trajectory.hpp"
#include <optional>

namespace rl {

auto KSpaceSingle(Trajectory const &traj, Re2 const &basis, float const bias = 1.f) -> Re2;

auto make_kspace_pre(
  std::string const &type, Index const nC, Trajectory const &traj, Re2 const &basis, float const bias)
  -> std::shared_ptr<Ops::Op<Cx>>;

} // namespace rl
