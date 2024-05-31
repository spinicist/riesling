#pragma once

#include "basis/basis.hpp"
#include "op/ops.hpp"
#include "trajectory.hpp"

#include <args.hxx>

namespace rl {

auto KSpaceSingle(
  Trajectory const &traj, Basis<Cx> const &basis, bool const vcc, float const bias = 1.f) -> Re2;

auto make_kspace_pre(Trajectory const  &traj,
                     Index const        nC,
                     Basis<Cx> const   &basis,
                     bool const         vcc,
                     std::string const &type = "kspace",
                     float const        bias = 1.f,
                     bool const         ndft = false) -> std::shared_ptr<Ops::Op<Cx>>;

} // namespace rl
