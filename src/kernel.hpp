#pragma once

#include "info.h"
#include <memory>

namespace rl {

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

  SizedKernel();

  int inPlane() const final;
  int throughPlane() const final;
  Sz3 dimensions() const;

  virtual KTensor k(Point3 const offset) const = 0;
  auto distSq(Point3 const p) const -> KTensor;

private:
  Eigen::TensorFixedSize<float, Eigen::Sizes<IP>> indIP, indTP;
};

std::unique_ptr<Kernel> make_kernel(std::string const &k, bool const is3D, float const os);

} // namespace rl
