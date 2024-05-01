#pragma once

#include "basis/basis.hpp"
#include "io/hd5.hpp"
#include "log.hpp"
#include "op/nufft.hpp"
#include "trajectory.hpp"

namespace rl {
namespace SENSE {

struct Opts
{
  Opts(args::Subparser &parser);
  args::ValueFlag<std::string>                   type;
  args::ValueFlag<Index>                         volume;
  args::ValueFlag<Eigen::Array3f, Array3fReader> res, fov;
  args::ValueFlag<float>                         λ;
  /*args::ValueFlag<Index>                         kRad, calRad, gap;
    args::ValueFlag<float>                         threshold;*/
};

//! Convenience function to get low resolution multi-channel images
auto LoresChannels(
  Opts &opts, GridOpts &gridOpts, Trajectory const &inTraj, Cx5 const &noncart, Basis<Cx> const &basis = IdBasis()) -> Cx5;

//! Normalizes by RSS with optional regularization
void RegularizedNormalization(float const λ, Cx4 const &ref, Cx5 &channels);
void RegularizedNormalization(float const λ, Cx5 &channels);

//! Convenience function called from recon commands to get SENSE maps
auto Choose(Opts &opts, GridOpts &gridOpts, Trajectory const &t, Cx5 const &noncart) -> Cx5;

} // namespace SENSE
} // namespace rl
