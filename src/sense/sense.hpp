#pragma once

#include "basis/basis.hpp"
#include "io/hd5.hpp"
#include "log.hpp"
#include "parse_args.hpp"
#include "trajectory.hpp"

namespace rl {
namespace SENSE {

struct Opts
{
  Opts(args::Subparser &parser);
  args::ValueFlag<std::string>                   type;
  args::ValueFlag<Index>                         volume;
  args::ValueFlag<float>                         res, λ;
  args::ValueFlag<Eigen::Array3f, Array3fReader> fov;
  args::ValueFlag<Index>                         kRad, calRad, gap;
  args::ValueFlag<float>                         threshold;
};

//! Convenience function to get low resolution multi-channel images
auto LoresChannels(Opts &opts, CoreOpts &coreOpts, Trajectory const &inTraj, Cx5 const &noncart, Cx2 const &basis = IdBasis())
  -> Cx5;

//! Normalizes by RSS with optional regularization
auto UniformNoise(float const λ, Sz3 const shape, Cx5 const &channels) -> Cx5;

//! Convenience function called from recon commands to get SENSE maps
auto Choose(Opts &opts, CoreOpts &core, Trajectory const &t, Cx5 const &noncart) -> Cx5;

} // namespace SENSE
} // namespace rl
