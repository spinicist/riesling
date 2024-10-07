#pragma once

#include "basis/basis.hpp"
#include "op/top.hpp"
#include "trajectory.hpp"

#include <args.hxx>

namespace rl {

auto KSpaceSingle(Trajectory const &traj, Basis::CPtr basis, bool const vcc, float const bias = 1.f) -> Re2;

auto MakeKspacePre(Trajectory const  &traj,
                   Index const        nC,
                   Index const        nS,
                   Index const        nT,
                   Basis::CPtr        basis,
                   std::string const &type = "kspace",
                   float const        bias = 1.f) -> TOps::TOp<Cx, 5, 5>::Ptr;

} // namespace rl
