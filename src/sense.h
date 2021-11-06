#pragma once

#include "cropper.h"
#include "info.h"
#include "io.h"
#include "log.h"
#include "op/grid.h"
#include "parse_args.h"

#define COMMON_SENSE_ARGS                                                                          \
  args::ValueFlag<std::string> senseFile(                                                          \
    parser, "SENSE", "Read SENSE maps from specified .h5 file", {"sense", 's'});                   \
  args::ValueFlag<long> senseVol(                                                               \
    parser, "SENSE VOLUME", "Take SENSE maps from this volume", {"senseVolume"}, -1);              \
  args::ValueFlag<float> senseLambda(                                                              \
    parser, "LAMBDA", "SENSE regularization", {"lambda", 'l'}, 0.f);

// Helper function for getting a good volume to take SENSE maps from
long ValOrLast(long const val, long const last);

/*!
 * Calculates a set of SENSE maps from non-cartesian data, assuming an oversampled central region
 */
Cx4 DirectSENSE(
  Trajectory const &traj,
  float const os,
  bool const kb,
  float const fov,
  float const lambda,
  long const volume,
  HD5::Reader &reader,
  Log &log);

/*!
 * Loads a set of SENSE maps from a file
 */
Cx4 LoadSENSE(std::string const &calFile, Log &log);

/*!
 * Loads a set of SENSE maps from a file and interpolate them to correct dims
 */
Cx4 InterpSENSE(std::string const &file, Eigen::Array3l const dims, Log &log);
