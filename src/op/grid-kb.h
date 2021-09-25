#pragma once

#include "../fft_plan.h"
#include "../trajectory.h"
#include "grid.h"

template <int InPlane, int ThroughPlane>
struct GridKB final : GridOp
{
  GridKB(
      Trajectory const &traj,
      float const os,
      bool const unsafe,
      Log &log,
      float const inRes = -1.f,
      bool const shrink = false);

  void A(Input const &x, Output &y) const;
  void Adj(Output const &x, Input &y) const;

  R3 apodization(Sz3 const sz) const;

private:
  using InPlaneArray = Eigen::TensorFixedSize<float, Eigen::Sizes<InPlane>>;
  using ThroughPlaneArray = Eigen::TensorFixedSize<float, Eigen::Sizes<ThroughPlane>>;
  using Kernel = Eigen::TensorFixedSize<float, Eigen::Sizes<InPlane, InPlane, ThroughPlane>>;

  float betaIn_, betaThrough_;
  InPlaneArray indIn_;
  ThroughPlaneArray indThrough_;
  void kernel(Point3 const offset, float const dc, Kernel &k) const;
  FFT::ThreeD fft_; // For sqrt kernel
};

using GridKB3D = GridKB<3, 3>;
using GridKB2D = GridKB<3, 1>;
