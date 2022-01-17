#pragma once

#include "info.h"
#include <memory>

struct Kernel
{
  virtual ~Kernel(){};
  virtual int inPlane() const = 0;
  virtual int throughPlane() const = 0;
};

template <int IP, int TP>
struct SizedKernel : Kernel
{
  using KTensor = Eigen::TensorFixedSize<float, Eigen::Sizes<IP, IP, TP>>;
  static Index const InPlane = IP;
  static Index const ThroughPlane = TP;

  int inPlane() const final
  {
    return InPlane;
  }
  int throughPlane() const final
  {
    return ThroughPlane;
  }

  virtual KTensor k(Point3 const offset) const = 0;
  KTensor distSq(Point3 const p) const;
};

std::unique_ptr<Kernel> make_kernel(std::string const &k, Info::Type const t, float const os);