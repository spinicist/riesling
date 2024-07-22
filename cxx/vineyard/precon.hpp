#pragma once

#include "basis/basis.hpp"
#include "op/ops.hpp"
#include "trajectory.hpp"

#include <args.hxx>

namespace rl {

auto KSpaceSingle(Trajectory const &traj, Basis::CPtr basis, bool const vcc, float const bias = 1.f) -> Re2;

auto MakeKspacePre(Trajectory const  &traj,
                   Index const        nC,
                   Index const        nT,
                   Basis::CPtr        basis,
                   std::string const &type = "kspace",
                   float const        bias = 1.f,
                   bool const         ndft = false) -> std::shared_ptr<Ops::Op<Cx>>;

} // namespace rl
