#pragma once

#include "log.h"
#include "types.h"

/* This function is ported from ismrmd_phantom.cpp, which in turn is inspired by
 * http://web.eecs.umich.edu/~fessler/code/
 */
Cx4 birdcage(
    Eigen::Array3l const &matrix,
    Eigen::Array3f const &voxel_size,
    long const channels,
    long const nrings,
    float const coil_rad_mm,  // Radius of the actual coil, i.e. where the channels should go
    float const sense_rad_mm, // Sensitivity radius
    Log const &log);