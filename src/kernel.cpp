#include "kernel.hpp"

#include "kernel-fi.hpp"
#include "kernel-kb.hpp"
#include "kernel-nn.hpp"

#include "log.h"

namespace rl {

template <int IP, int TP>
SizedKernel<IP, TP>::SizedKernel()
{
  // Note INTEGER division
  std::iota(indIP.data(), indIP.data() + IP, -IP / 2);
  std::iota(indTP.data(), indTP.data() + TP, -TP / 2);
}

template <int IP, int TP>
auto SizedKernel<IP, TP>::inPlane() const -> int
{
  return InPlane;
}

template <int IP, int TP>
auto SizedKernel<IP, TP>::throughPlane() const -> int
{
  return ThroughPlane;
}

template <int IP, int TP>
auto SizedKernel<IP, TP>::dimensions() const -> Sz3
{
  return Sz3{InPlane, InPlane, ThroughPlane};
}

template <int IP, int TP>
auto SizedKernel<IP, TP>::distSq(Point3 const p) const -> KTensor
{
  // Yes, below is a weird mix of integer and floating point division.
  // But it works
  KTensor k;
  constexpr float IP_2 = IP / 2.f;
  constexpr Index IIP_2 = IP / 2;
  if constexpr (TP > 1) {
    float const TP_2 = TP / 2.f;
    Index const ITP_2 = TP / 2;
    Point3 const np = p.array() / Point3(IP_2, IP_2, TP_2).array();
    for (Index iz = 0; iz < TP; iz++) {
      for (Index iy = 0; iy < IP; iy++) {
        for (Index ix = 0; ix < IP; ix++) {
          k(ix, iy, iz) = (np - Point3((-IIP_2 + ix) / IP_2, (-IIP_2 + iy) / IP_2, (-ITP_2 + iz) / TP_2)).squaredNorm();
        }
      }
    }
  } else {
    Point3 const np = p.array() / Point3(IP_2, IP_2, 1.f).array();
    for (Index iy = 0; iy < IP; iy++) {
      for (Index ix = 0; ix < IP; ix++) {
        k(ix, iy, 0) = (np - Point3((-IIP_2 + ix) / IP_2, (-IIP_2 + iy) / IP_2, 0.f)).squaredNorm();
      }
    }
  }
  return k;
}

template struct SizedKernel<1, 1>;
template struct SizedKernel<3, 1>;
template struct SizedKernel<3, 3>;
template struct SizedKernel<5, 1>;
template struct SizedKernel<5, 5>;
template struct SizedKernel<7, 1>;
template struct SizedKernel<7, 7>;

std::unique_ptr<Kernel> make_kernel(std::string const &k, Info::Type const t, float const os)
{
  if (k == "NN") {
    return std::make_unique<NearestNeighbour>();
  } else if (k == "KB3") {
    if (t == Info::Type::ThreeD) {
      return std::make_unique<KaiserBessel<3, 3>>(os);
    } else {
      return std::make_unique<KaiserBessel<3, 1>>(os);
    }
  } else if (k == "KB5") {
    if (t == Info::Type::ThreeD) {
      return std::make_unique<KaiserBessel<5, 5>>(os);
    } else {
      return std::make_unique<KaiserBessel<5, 1>>(os);
    }
  } else if (k == "KB7") {
    if (t == Info::Type::ThreeD) {
      return std::make_unique<KaiserBessel<7, 7>>(os);
    } else {
      return std::make_unique<KaiserBessel<7, 1>>(os);
    }
  } else if (k == "FI3") {
    if (t == Info::Type::ThreeD) {
      return std::make_unique<FlatIron<3, 3>>(os);
    } else {
      return std::make_unique<FlatIron<3, 1>>(os);
    }
  } else if (k == "FI5") {
    if (t == Info::Type::ThreeD) {
      return std::make_unique<FlatIron<5, 5>>(os);
    } else {
      return std::make_unique<FlatIron<5, 1>>(os);
    }
  } else if (k == "FI7") {
    if (t == Info::Type::ThreeD) {
      return std::make_unique<FlatIron<7, 7>>(os);
    } else {
      return std::make_unique<FlatIron<7, 1>>(os);
    }
  } else {
    Log::Fail(FMT_STRING("Unknown kernel type: {}"), k);
  }
}

} // namespace rl
