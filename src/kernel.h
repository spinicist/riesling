#pragma once

#include "fft_plan.h"
#include "types.h"

template <int InPlane_, int ThroughPlane_>
struct KaiserBessel
{
  constexpr static int InPlane = InPlane_;
  constexpr static int ThroughPlane = ThroughPlane_;
  using KTensor = Eigen::TensorFixedSize<float, Eigen::Sizes<InPlane, InPlane, ThroughPlane>>;
  KaiserBessel(float os);

  KTensor operator()(Point3 const offset) const; // This expects x to already be squared

private:
  float beta_, scale_;
};

template <int InPlane_, int ThroughPlane_>
struct PipeSDC
{
  constexpr static int InPlane = InPlane_;
  constexpr static int ThroughPlane = ThroughPlane_;
  using KTensor = Eigen::TensorFixedSize<float, Eigen::Sizes<InPlane, InPlane, ThroughPlane>>;
  PipeSDC(float os);

  KTensor operator()(Point3 const offset) const;

private:
  float distScale_, valScale_;
};

enum struct Kernels
{
  NN = 0,
  KB3 = 1,
  KB5 = 2
};
