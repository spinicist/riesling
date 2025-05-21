#pragma once

#include "../basis/basis.hpp"
#include "../io/hd5.hpp"
#include "../log.hpp"
#include "../op/grid-opts.hpp"
#include "../trajectory.hpp"

namespace rl {
namespace SENSE {

struct Opts
{
  std::string    type;
  Index          tp, kWidth;
  Eigen::Array3f res;
  float          l, λ;
};

//! Convenience function to get low resolution multi-channel images
auto LoresChannels(
  Opts const &opts, GridOpts<3> const &gridOpts, Trajectory traj, Cx5 const &noncart, Basis::CPtr basis = nullptr) -> Cx5;

auto TikhonovDivision(Cx5 const &channels, Cx4 const &ref, float const λ) -> Cx5;
auto EstimateMaps(Cx5 const &ichan, Cx4 const &ref, float const os, float const l, float const λ) -> Cx5;
auto EstimateKernels(Cx5 const &nomChan, Cx4 const &nomRef, Index const nomKW, float const osamp, float const l, float const λ)
  -> Cx5;
template <int ND> auto KernelsToMaps(CxN<ND + 2> const &kernels, Sz<ND> const mat, float const os) -> CxN<ND + 2>;
auto                   MapsToKernels(Cx5 const &maps, Index const kW, float const os) -> Cx5;

//! Convenience function called from recon commands to get SENSE maps
auto Choose(Opts const &opts, GridOpts<3> const &gridOpts, Trajectory const &t, Cx5 const &noncart) -> Cx5;

} // namespace SENSE
} // namespace rl
