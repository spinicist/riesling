#pragma once

#include "log.h"
#include "trajectory.h"
#include "types.h"
#include <unordered_map>

namespace SDC {
R2 Pipe(Trajectory const &traj, Log &log);
R2 Radial(Trajectory const &traj, Log &log);
R2 Choose(std::string const &fname, Trajectory const &traj, Log &log);
} // namespace SDC
