#pragma once

#include "info.h"
#include "log.h"
#include "types.h"
#include <unordered_map>

// Forward declare
struct Trajectory;
struct GridOp;
struct GridBasisOp;

namespace SDC {

void Choose(
    std::string const &fname, Trajectory const &traj, std::unique_ptr<GridOp> &gridder, Log &log);
void Choose(
    std::string const &fname,
    Trajectory const &traj,
    std::unique_ptr<GridOp> &gridder,
    std::unique_ptr<GridBasisOp> &gridder2,
    Log &log);

R2 Pipe(Trajectory const &traj, std::unique_ptr<GridOp> &gridder, Log &log);
R2 Radial(Trajectory const &traj, Log &log);
} // namespace SDC
