#pragma once

#include "cropper.h"
#include "gridder.h"
#include "info.h"
#include "io_hd5.h"
#include "log.h"

/*!
 * Calculates a set of SENSE maps from non-cartesian data, assuming an oversampled central region
 */
Cx4 DirectSENSE(
    Trajectory const &traj,
    float const os,
    Kernel *kernel,
    float const fov,
    Cx3 const &data,
    float const lambda,
    Log &log);

Cx4 LoadSENSE(std::string const &calFile, Sz4 const dims, Log &log);
