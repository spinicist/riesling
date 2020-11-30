#pragma once

#include "log.h"
#include "radial.h"

/*!
 * Calculates a set of SENSE maps from radial data, assuming an oversampled central region
 */
Cx4 SENSE(RadialInfo const &info, R3 const &traj, float const os, Cx3 const &radial, Log &log);
