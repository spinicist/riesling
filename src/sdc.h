#pragma once

#include "info.h"
#include "log.h"
#include "types.h"
#include <unordered_map>

// Forward declare
struct Gridder;
struct Kernel;

namespace SDC {

void Load(
    std::string const &fname,
    Info const &info,
    R3 const &traj,
    Kernel *kernel,
    Gridder &gridder,
    Log &log);
R2 Pipe(Info const &info, Gridder &gridder, Kernel *kernel, Log &log);
R2 Radial(Info const &info, R3 const &traj, Log &log);
} // namespace SDC
