#pragma once

#include "io/io.h"
#include "log.h"
#include "op/grids.h"
#include "op/sdc.hpp"
#include "parse_args.h"
#include "trajectory.h"

#define COMMON_SENSE_ARGS                                                                          \
  args::ValueFlag<std::string> sFile(parser, "F", "Read SENSE maps from .h5", {"sense", 's'});     \
  args::ValueFlag<Index> sVol(parser, "V", "SENSE calibration volume", {"senseVolume"}, -1);       \
  args::ValueFlag<float> sRes(parser, "R", "SENSE calibration res (12 mm)", {"senseRes"}, 12.f);   \
  args::ValueFlag<float> sReg(parser, "L", "SENSE regularization", {"senseReg"}, 0.f);

namespace SENSE {

//! Calculates a set of SENSE maps from non-cartesian data, assuming an oversampled central region
Cx4 SelfCalibration(
  Info const &info,
  GridBase *g,
  float const fov,
  float const res,
  float const reg,
  Cx3 const &data);
Cx4 Load(std::string const &calFile); //! Loads a set of SENSE maps from a file
Cx4 Interp(std::string const &calFile, Eigen::Array3l const dims); //! Interpolate with FFT

//! Convenience function called from recon commands to get SENSE maps
Cx4 Choose(
  std::string const &path,
  Info const &i,
  GridBase *g,
  float const fov,
  float const res,
  float const reg,
  SDCOp *sdc,
  Cx3 const &data);

} // namespace SENSE
