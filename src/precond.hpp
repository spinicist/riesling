#pragma once

#include "op/tensorscale.hpp"
#include "trajectory.hpp"
#include <optional>

namespace rl {

auto KSpaceSingle(Trajectory const &traj, std::optional<Re2> const &basis = std::nullopt, float const bias = 1.f) -> Re2;

std::shared_ptr<TensorOperator<Cx, 4>> make_kspace_pre(
  std::string const &type, Sz4 const dims, Trajectory const &traj, std::optional<Re2> const &basis, float const bias);

} // namespace rl
