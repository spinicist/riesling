#pragma once

#include "cropper.h"
#include "gridder.h"
#include "info.h"
#include "io_hd5.h"
#include "log.h"

/*!
 * Calculates a set of SENSE maps from non-cartesian data, assuming an oversampled central region
 */
Cx4 DirectSENSE(Gridder const &gridder, Cx3 const &data, float const lambda, Log &log);

Cx4 LoadSENSE(
    long const nChan,
    Cropper const &cropper,
    std::string const &calFile,
    long const calVolume,
    HD5::Reader &reader,
    Trajectory const &traj,
    float const os,
    Kernel *kernel,
    float const lambda,
    Cx3 &ks,
    long &currentVol,
    Log &log);
