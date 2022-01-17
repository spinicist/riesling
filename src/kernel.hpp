#pragma once

#include "kernel.h"
#include "tensorOps.h"

template <int IP, int TP>
auto SizedKernel<IP, TP>::distSq(Point3 const p) const -> KTensor
{
  using FixIP = Eigen::type2index<IP>;
  using FixTP = Eigen::type2index<TP>;
  Eigen::TensorFixedSize<float, Eigen::Sizes<IP>> indIP;
  Eigen::TensorFixedSize<float, Eigen::Sizes<TP>> indTP;
  std::iota(indIP.data(), indIP.data() + IP, -IP / 2); // Note INTEGER division
  std::iota(indTP.data(), indTP.data() + TP, -TP / 2); // Note INTEGER division
  constexpr Eigen::IndexList<FixIP, FixOne, FixOne> rshX;
  constexpr Eigen::IndexList<FixOne, FixIP, FixTP> brdX;
  constexpr Eigen::IndexList<FixOne, FixIP, FixOne> rshY;
  constexpr Eigen::IndexList<FixIP, FixOne, FixTP> brdY;
  constexpr Eigen::IndexList<FixOne, FixOne, FixTP> rshZ;
  constexpr Eigen::IndexList<FixIP, FixIP, FixOne> brdZ;
  auto const kx = (indIP.constant(p[0]) - indIP).square().reshape(rshX).broadcast(brdX);
  auto const ky = (indIP.constant(p[1]) - indIP).square().reshape(rshY).broadcast(brdY);
  auto const kz = (indTP.constant(p[2]) - indTP).square().reshape(rshZ).broadcast(brdZ);
  return kx + ky + kz;
}

struct NearestNeighbour final : SizedKernel<1, 1>
{
  using typename SizedKernel<1, 1>::KTensor;

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
  }

  KTensor k(Point3 const p) const
  {
    constexpr float W_2 = (IP / 2.f) * (IP / 2.f);
    auto const x = SizedKernel<IP, TP>::distSq(p);
    return (x < W_2).select(
      x.constant(scale_) *
        (x.constant(beta_) * (x.constant(1.f) - (x / x.constant(W_2))).sqrt()).bessel_i0(),
      x.constant(0.f));
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
  }

  KTensor k(Point3 const p) const
  {
    auto const x = SizedKernel<IP, TP>::distSq(p);
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

private:
  float distScale_, valScale_;
};
