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

  SizedKernel()
  {
    // Note INTEGER division
    std::iota(indIP.data(), indIP.data() + IP, -IP / 2);
    std::iota(indTP.data(), indTP.data() + TP, -TP / 2);
  }

  int inPlane() const final
  {
    return InPlane;
  }
  int throughPlane() const final
  {
    return ThroughPlane;
  }
  Sz3 dimensions() const
  {
    return Sz3{InPlane, InPlane, ThroughPlane};
  }

  virtual KTensor k(Point3 const offset) const = 0;
  virtual KTensor distSq(Point3 const p) const;

private:
  Eigen::TensorFixedSize<float, Eigen::Sizes<IP>> indIP, indTP;
};

std::unique_ptr<Kernel> make_kernel(std::string const &k, Info::Type const t, float const os);

} // namespace rl
