#pragma once

#include "cropper.h"
#include "info.h"
#include "io_hd5.h"
#include "log.h"
#include "op/grid.h"
#include "parse_args.h"

#define COMMON_SENSE_ARGS                                                                          \
  args::ValueFlag<std::string> senseFile(                                                          \
      parser, "SENSE", "Read SENSE maps from specified .h5 file", {"sense", 's'});                 \
  args::ValueFlag<long> senseVolume(                                                               \
      parser, "SENSE VOLUME", "Take SENSE maps from this volume", {"senseVolume"}, 0);             \
  args::ValueFlag<float> senseLambda(                                                              \
      parser, "LAMBDA", "SENSE regularization", {"lambda", 'l'}, 0.f);

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
 * Loads a set of SENSE maps from a file
 */
Cx4 LoadSENSE(std::string const &calFile, Log &log);

/*!
 * Loads a set of SENSE maps from a file and interpolate them to correct dims
 */
Cx4 InterpSENSE(std::string const &file, Eigen::Array3l const dims, Log &log);
