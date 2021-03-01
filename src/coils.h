#pragma once

#include "info.h"
#include "log.h"
#include "types.h"

/* This function is ported from ismrmd_phantom.cpp, which in turn is inspired by
 * http://web.eecs.umich.edu/~fessler/code/
 */
Cx4 birdcage(
    Dims3 const sz,
    long const nchan,
    float const coil_rad_mm,  // Radius of the actual coil, i.e. where the channels should go
    float const sense_rad_mm, // Sensitivity radius
    Info const &info,
    Log const &log);