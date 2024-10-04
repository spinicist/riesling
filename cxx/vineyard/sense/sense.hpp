#pragma once

#include "basis/basis.hpp"
#include "io/hd5.hpp"
#include "log.hpp"
#include "op/grid.hpp"
#include "sys/args.hpp"
#include "trajectory.hpp"

namespace rl {
namespace SENSE {

struct Opts
{
  Opts(args::Subparser &parser);
  args::ValueFlag<std::string>                   type;
  args::ValueFlag<Index>                         volume, kWidth;
  args::ValueFlag<Eigen::Array3f, Array3fReader> res, fov;
  args::ValueFlag<float>                         λ;
  args::Flag                                     decant;
};

//! Convenience function to get low resolution multi-channel images
auto LoresChannels(Opts &opts, GridOpts &gridOpts, Trajectory const &inTraj, Cx5 const &noncart, Basis::CPtr basis = nullptr)
  -> Cx5;
auto LoresKernels(Opts &opts, GridOpts &gridOpts, Trajectory const &inTraj, Cx5 const &noncart, Basis::CPtr basis = nullptr)
  -> Cx5;

auto TikhonovDivision(Cx5 const &channels, Cx4 const &ref, float const λ) -> Cx5;
auto EstimateKernels(Cx5 const &channels, Cx4 const &ref, Index const kW, float const λ) -> Cx5;
auto KernelsToMaps(Cx5 const &kernels, Sz3 const fmat, Sz3 const cmat) -> Cx5;

//! Convenience function called from recon commands to get SENSE maps
auto Choose(Opts &opts, GridOpts &gridOpts, Trajectory const &t, Cx5 const &noncart) -> Cx5;

} // namespace SENSE
} // namespace rl
