#include "kernel.h"

#include "tensorOps.h"

template <int IP, int TP>
Eigen::TensorFixedSize<float, Eigen::Sizes<IP, IP, TP>> DistSq(Point3 const p)
{
  using KTensor = Eigen::TensorFixedSize<float, Eigen::Sizes<IP, IP, TP>>;
  using KArray = Eigen::TensorFixedSize<float, Eigen::Sizes<IP>>;
  using FixIn = Eigen::type2index<IP>;
  KArray indices;
  std::iota(indices.data(), indices.data() + IP, -IP / 2); // Note INTEGER division
  KTensor k;
  if constexpr (TP > 1) {
    constexpr Eigen::IndexList<FixIn, FixOne, FixOne> rshX;
    constexpr Eigen::IndexList<FixOne, FixIn, FixIn> brdX;
    constexpr Eigen::IndexList<FixOne, FixIn, FixOne> rshY;
    constexpr Eigen::IndexList<FixIn, FixOne, FixIn> brdY;
    constexpr Eigen::IndexList<FixOne, FixOne, FixIn> rshZ;
    constexpr Eigen::IndexList<FixIn, FixIn, FixOne> brdZ;
    auto const kx = (indices.constant(p[0]) - indices).square().reshape(rshX).broadcast(brdX);
    auto const ky = (indices.constant(p[1]) - indices).square().reshape(rshY).broadcast(brdY);
    auto const kz = (indices.constant(p[2]) - indices).square().reshape(rshZ).broadcast(brdZ);
    k = kx + ky + kz;
  } else {
    constexpr Eigen::IndexList<FixIn, FixOne, FixOne> rshX;
    constexpr Eigen::IndexList<FixOne, FixIn, FixOne> brdX;
    constexpr Eigen::IndexList<FixOne, FixIn, FixOne> rshY;
    constexpr Eigen::IndexList<FixIn, FixOne, FixOne> brdY;
    auto const kx = (indices.constant(p[0]) - indices).square().reshape(rshX).broadcast(brdX);
    auto const ky = (indices.constant(p[1]) - indices).square().reshape(rshY).broadcast(brdY);
    k = kx + ky;
  }
  return k;
}

template <int IP, int TP>
KaiserBessel<IP, TP>::KaiserBessel(float const os)
  : beta_{(float)M_PI * sqrtf(pow(IP * (os - 0.5f) / os, 2.f) - 0.8f)}
{
  // Get the normalization factor
  scale_ = 1.f;
  KTensor k = operator()(Point3::Zero());
  scale_ = 1.f / Sum(k);
}

template <int IP, int TP>
auto KaiserBessel<IP, TP>::operator()(Point3 const p) const -> KTensor
{
  constexpr float W_2 = (IP / 2.f) * (IP / 2.f);
  KTensor x = DistSq<IP, TP>(p);
  return (x < W_2).select(
    x.constant(scale_) *
      (x.constant(beta_) * (x.constant(1.f) - (x / x.constant(W_2))).sqrt()).bessel_i0(),
    x.constant(0.f));
}

template <int IP, int TP>
PipeSDC<IP, TP>::PipeSDC(float const os)
  : distScale_{std::pow(0.96960938f * 25600.f / (os * 63.f), 2.f)}
// Magic numbers from Pipe's code
{
  valScale_ = 1.f;
  KTensor k = operator()(Point3::Zero());
  valScale_ = 1.f / Sum(k);
}

template <int IP, int TP>
typename PipeSDC<IP, TP>::KTensor PipeSDC<IP, TP>::operator()(Point3 const p) const
{
  KTensor x = DistSq<IP, TP>(p);
  KTensor x2 = (x * x.constant(distScale_)).sqrt(); // Pipe code unclear if sq or not
  constexpr float c = 0.99992359966186584;          // Constant term
  constexpr std::array<double, 5> coeffs{
    3.4511129626832091E-05,
    -1.7986635886194154E-05,
    1.3282009203652969E-08,
    8.5313956268885989E-11,
    -1.1469041640943728E-13};

  KTensor result;
  result.setConstant(c);
  for (size_t ii = 0; ii < coeffs.size(); ii++) {
    result += x2.constant(coeffs[ii]) * x2.pow(ii + 1);
  }
  result = (result > 0.f).select(result * result.constant(valScale_), result.constant(0.f));
  // fmt::print("p {}\n{}\n{}\n", p.transpose(), x, result);
  return result;
}

template struct KaiserBessel<3, 1>;
template struct KaiserBessel<3, 3>;
template struct KaiserBessel<5, 1>;
template struct KaiserBessel<5, 5>;
template struct PipeSDC<5, 5>;
template struct PipeSDC<5, 1>;
