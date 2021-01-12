#pragma once

#include "info.h"
#include "log.h"

/*!
 * Calculates a set of SENSE maps from non-cartesian data, assuming an oversampled central region
 */
Cx4 SENSE(
    Info const &info,
    R3 const &traj,
    float const os,
    bool const stack,
    bool const kb,
    Cx3 const &data,
    Log &log);

Cx4 EigenSENSE(
    Info const &info,
    R3 const &traj,
    float const os,
    bool const stack,
    bool const kb,
    long const nc,
    Cx3 const &data,
    Log &log);
