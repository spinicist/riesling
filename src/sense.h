#pragma once

#include "gridder.h"
#include "info.h"
#include "log.h"

/*!
 * Calculates a set of SENSE maps from non-cartesian data, assuming an oversampled central region
 */
Cx4 SENSE(
    std::string const &method, Trajectory const &t, Gridder const &g, Cx3 const &data, Log &log);
