#pragma once

#include "op/ops.hpp"
#include "trajectory.hpp"
#include "basis.hpp"
#include <optional>

namespace rl {

auto KSpaceSingle(Trajectory const &traj, Cx2 const &basis = IdBasis(), float const bias = 1.f, bool const ndft = false) -> Re2;

auto make_kspace_pre(std::string const &type,
                     Index const        nC,
                     Trajectory const  &traj,
                     Cx2 const         &basis,
                     float const        bias = 1.f,
                     bool const         ndft = false) -> std::shared_ptr<Ops::Op<Cx>>;

} // namespace rl
