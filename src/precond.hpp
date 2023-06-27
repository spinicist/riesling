#pragma once

#include "op/ops.hpp"
#include "trajectory.hpp"
#include <optional>

namespace rl {

auto KSpaceSingle(Trajectory const &traj, std::optional<Re2> const &basis = std::nullopt) -> Re2;

auto make_kspace_pre(
  std::string const &type, Index const nC, Trajectory const &traj, std::optional<Re2> const &basis)
  -> std::shared_ptr<Ops::Op<Cx>>;

} // namespace rl
