#pragma once

#include "log.h"
#include "op/sdc.hpp"
#include "trajectory.h"
#include "types.h"

namespace SDC {
R2 Pipe(Trajectory const &traj, bool const nn, float const os, Index const max_its = 40);
R2 Radial(Trajectory const &traj, Index const lores, Index const gap);
std::unique_ptr<SDCOp>
Choose(std::string const &fname, Trajectory const &t, float const os, float const p = 1.f);
} // namespace SDC
