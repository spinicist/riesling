#pragma once

#include "kernel.h"
#include "log.h"
#include "tensorOps.h"

namespace rl {

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

struct NearestNeighbour final : SizedKernel<1, 1>
{
  using typename SizedKernel<1, 1>::KTensor;

  NearestNeighbour()
  {
    Log::Debug("Nearest-neighbour kernel");
  }

  KTensor k(Point3 const offset) const
  {
    KTensor k;
    k.setConstant(1.f);
    return k;
  }
};

template <int IP, int TP>
struct KaiserBessel final : SizedKernel<IP, TP>
{
  using typename SizedKernel<IP, TP>::KTensor;

  KaiserBessel(float os)
    : beta_{(float)M_PI * sqrtf(pow(IP * (os - 0.5f) / os, 2.f) - 0.8f)}
  {
    // Get the normalization factor
    scale_ = 1.f;
    scale_ = 1.f / Sum(k(Point3::Zero()));
    Log::Debug(FMT_STRING("Kaiser-Bessel kernel <{},{}> β={} scale={} "), IP, TP, beta_, scale_);
  }

  KTensor k(Point3 const p) const
  {
    auto const z2 = this->distSq(p);
    return (z2 > 1.f).select(
      z2.constant(0.f), z2.constant(scale_) * (z2.constant(beta_) * (z2.constant(1.f) - z2).sqrt()).bessel_i0());
  }

private:
  float beta_, scale_;
};

template <int IP, int TP>
struct FlatIron final : SizedKernel<IP, TP>
{
  using typename SizedKernel<IP, TP>::KTensor;

  FlatIron(float os)
    : beta_{(float)M_PI * 0.98f * IP * (1.f - 0.5f / os)}
  {
    // Get the normalization factor
    scale_ = 1.f;
    scale_ = 1.f / Sum(k(Point3::Zero()));
    Log::Debug(FMT_STRING("Flat Iron kernel <{},{}> β={}, scale={}"), IP, TP, beta_, scale_);
  }

  KTensor k(Point3 const p) const
  {
    auto const z2 = this->distSq(p);
    return (z2 > 1.f).select(
      z2.constant(0.f), z2.constant(scale_) * ((z2.constant(1.f) - z2).sqrt() * z2.constant(beta_)).exp());
  }

private:
  float beta_, scale_;
};

template <int IP, int TP>
struct PipeSDC final : SizedKernel<IP, TP>
{
  using typename SizedKernel<IP, TP>::KTensor;

  PipeSDC(float os)
    : distScale_{std::pow(0.96960938f * 25600.f / (os * 63.f), 2.f)}
  // Magic numbers from Pipe's code
  {
    valScale_ = 1.f;
    valScale_ = 1.f / Sum(k(Point3::Zero()));
    Log::Debug(FMT_STRING("Pipe/Zwart kernel <{},{}> β={}, scale={}"), IP, TP, distScale_, valScale_);
  }

  KTensor k(Point3 const p) const
  {
    auto const x = distSq(p);
    auto const x2 = (x * x.constant(distScale_)).sqrt(); // Pipe code unclear if sq or not
    constexpr float c = 0.99992359966186584;             // Constant term
    constexpr std::array<double, 5> coeffs{
      3.4511129626832091E-05,
      -1.7986635886194154E-05,
      1.3282009203652969E-08,
      8.5313956268885989E-11,
      -1.1469041640943728E-13};

    Eigen::TensorFixedSize<float, Eigen::Sizes<IP, IP, TP>> result;
    result.setConstant(c);
    for (size_t ii = 0; ii < coeffs.size(); ii++) {
      result += x2.constant(coeffs[ii]) * x2.pow(ii + 1);
    }
    result = (result > 0.f).select(result * result.constant(valScale_), result.constant(0.f));
    // fmt::print("p {}\n{}\n{}\n", p.transpose(), x, result);
    return result;
  }

  // Pipe et al use a different (unscaled) convention
  KTensor distSq(Point3 const p) const
  {
    KTensor k;
    if constexpr (TP > 1) {
      for (Index iz = 0; iz < TP; iz++) {
        for (Index iy = 0; iy < IP; iy++) {
          for (Index ix = 0; ix < IP; ix++) {
            k(ix, iy, iz) = (p - Point3(ix - IP / 2, iy - IP / 2, iz - TP / 2)).squaredNorm();
          }
        }
      }
    } else {
      for (Index iy = 0; iy < IP; iy++) {
        for (Index ix = 0; ix < IP; ix++) {
          k(ix, iy, 0) = (p - Point3(ix - IP / 2, iy - IP / 2, 0.f)).squaredNorm();
        }
      }
    }
    return k;
  }

private:
  float distScale_, valScale_;
};

} // namespace rl
