#pragma once

#include "io/hd5.hpp"
#include "log.hpp"
#include "parse_args.hpp"
#include "trajectory.hpp"

namespace rl {
namespace SENSE {

struct Opts
{
  Opts(args::Subparser &parser);
  args::ValueFlag<std::string> type;
  args::ValueFlag<Index>       volume, frame;
  args::ValueFlag<float>       res, λ;
  args::ValueFlag<Eigen::Array3f, Array3fReader> fov;
  args::ValueFlag<Index>       kRad, calRad, gap;
  args::ValueFlag<float>       threshold;
};

// Normalizes by RSS with optional regularization
auto UniformNoise(float const λ, Sz3 const shape, Cx4 &channels) -> Cx4;

//! Convenience function called from recon commands to get SENSE maps
Cx4 Choose(Opts &opts, CoreOpts &core, Trajectory const &t, Cx5 const &noncart);

} // namespace SENSE
} // namespace rl
