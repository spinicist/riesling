#pragma once

#include "info.h"
#include "kernel.h"
#include "log.h"

/*!
 * Calculates a set of SENSE maps from non-cartesian data, assuming an oversampled central region
 */
Cx4 SENSE(
    Info const &info,
    R3 const &traj,
    float const os,
    Kernel *const kb,
    bool const shrink,
    std::string const &sdc,
    float const threshold,
    Cx3 const &data,
    Log &log);
