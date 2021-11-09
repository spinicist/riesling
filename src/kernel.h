#pragma once

#include "fft_plan.h"
#include "types.h"

template <int InPlane, int ThroughPlane>
struct Kernel
{
  using KTensor = Eigen::TensorFixedSize<float, Eigen::Sizes<InPlane, InPlane, ThroughPlane>>;

  Kernel(float const os);
  KTensor operator()(Point3 const offset, float const scale) const;

  void sqrtOn();
  void sqrtOff();

private:
  using KArray = Eigen::TensorFixedSize<float, Eigen::Sizes<InPlane>>;
  using FixIn = Eigen::type2index<InPlane>;
  using FixThrough = Eigen::type2index<ThroughPlane>;

  float beta_, kScale_;
  KArray indices_;
  FFT::ThreeD fft_; // For sqrt kernel
  bool sqrt_;
};