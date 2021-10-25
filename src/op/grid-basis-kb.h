#pragma once

#include "../fft_plan.h"
#include "../trajectory.h"
#include "grid-basis.h"

template <int InPlane, int ThroughPlane>
struct GridBasisKB final : GridBasisOp
{
  GridBasisKB(
      Trajectory const &traj,
      float const os,
      bool const unsafe,
      R2 &basis,
      Log &log,
      float const inRes = -1.f,
      bool const shrink = false);
  GridBasisKB(
      Mapping const &mapping,
      bool const unsafe,
      R2 &basis,
      Log &log);

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

using GridBasisKB3D = GridBasisKB<3, 3>;
using GridBasisKB2D = GridBasisKB<3, 1>;
