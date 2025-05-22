#pragma once

#include "../basis/basis.hpp"
#include "../io/hd5.hpp"
#include "../log.hpp"
#include "../op/grid-opts.hpp"
#include "../trajectory.hpp"

namespace rl {
namespace SENSE {

template <int ND> struct Opts
{
  std::string                type;
  Index                      tp, kWidth;
  Eigen::Array<float, ND, 1> res;
  float                      l, λ;
};

//! Convenience function to get low resolution multi-channel images
template <int ND> auto LoresChannels(
  Opts<ND> const &opts, GridOpts<ND> const &gridOpts, Trajectory traj, Cx5 const &noncart, Basis::CPtr basis = nullptr) -> Cx5;

auto TikhonovDivision(Cx5 const &channels, Cx4 const &ref, float const λ) -> Cx5;
template <int ND>
auto EstimateKernels(Cx5 const &nomChan, Cx4 const &nomRef, Index const nomKW, float const osamp, float const l, float const λ)
  -> Cx5;
template <int ND> auto KernelsToMaps(CxN<ND + 2> const &kernels, Sz<ND> const mat, float const os) -> CxN<ND + 2>;
auto                   MapsToKernels(Cx5 const &maps, Index const kW, float const os) -> Cx5;

//! Convenience function called from recon commands to get SENSE maps
template <int ND> auto Choose(Opts<ND> const &opts, GridOpts<ND> const &gridOpts, Trajectory const &t, Cx5 const &noncart)
  -> Cx5;

} // namespace SENSE
} // namespace rl
