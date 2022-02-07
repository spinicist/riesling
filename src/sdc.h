#pragma once

#include "log.h"
#include "precond/sdc.hpp"
#include "trajectory.h"
#include "types.h"

namespace SDC {
R2 Pipe(Trajectory const &traj, bool const nn, float const os);
R2 Radial(Trajectory const &traj, Index const lores, Index const gap);
SDCPrecond Choose(std::string const &fname, float const p, Trajectory const &t, float const os);
} // namespace SDC
