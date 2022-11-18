#pragma once

#include "io/hd5.hpp"
#include "log.hpp"
#include "op/make_grid.hpp"
#include "parse_args.hpp"
#include "trajectory.hpp"

namespace rl {
namespace SENSE {

struct Opts
{
  Opts(args::Subparser &parser);
  args::ValueFlag<std::string> file;
  args::ValueFlag<Index> volume, frame;
  args::ValueFlag<float> res, λ, fov;
};

//! Calculates a set of SENSE maps from non-cartesian data, assuming an oversampled central region
Cx4 SelfCalibration(Opts &opts, CoreOpts &coreOpts, Trajectory const &inTraj, HD5::Reader &reader);

//! Convenience function called from recon commands to get SENSE maps
Cx4 Choose(Opts &opts, CoreOpts &core, Trajectory const &t, HD5::Reader &reader);

} // namespace SENSE
} // namespace rl
