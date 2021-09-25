#pragma once

#include "cropper.h"
#include "info.h"
#include "io_hd5.h"
#include "log.h"
#include "op/grid.h"

/*!
 * Calculates a set of SENSE maps from non-cartesian data, assuming an oversampled central region
 */
Cx4 DirectSENSE(
    Trajectory const &traj,
    float const os,
    bool const kb,
    float const fov,
    Cx3 const &data,
    float const lambda,
    Log &log);

/*!
 * Loads a set of SENSE maps from a file, they must match dims
 */
Cx4 LoadSENSE(std::string const &calFile, Sz4 const dims, Log &log);

/*!
 * Loads a set of SENSE maps from a file and interpolate them to correct dims
 */
Cx4 InterpSENSE(std::string const &file, Eigen::Array3l const dims, Log &log);
