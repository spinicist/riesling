#include "kernel.h"

#include "tensorOps.h"

template <int InPlane, int ThroughPlane>
Kernel<InPlane, ThroughPlane>::Kernel(float const os)
  : beta_{(float)M_PI * sqrtf(pow(InPlane * (os - 0.5f) / os, 2.f) - 0.8f)}
  , kScale_{1.f}
  , fft_{Sz3{InPlane, InPlane, ThroughPlane}, Log(), 1}
  , sqrt_{false}
{
  // If 3D then must be spherically symmetric
  static_assert((ThroughPlane == 1) || (ThroughPlane == InPlane));
  // Set up arrays of indices for building kernel
  std::iota(indices_.data(), indices_.data() + InPlane, -InPlane / 2);
  auto k = operator()(Point3::Zero(), 1.f);
  kScale_ = 1.f / Sum(k);
}

template <int InPlane, int ThroughPlane>
void Kernel<InPlane, ThroughPlane>::sqrtOn()
{
  sqrt_ = true;
}

template <int InPlane, int ThroughPlane>
void Kernel<InPlane, ThroughPlane>::sqrtOff()
{
  sqrt_ = false;
}

// This expects x to already be squared
template <int W, typename T>
inline decltype(auto) KB(T const &x, float const beta, float const scale)
{
  constexpr float W_2 = (W / 2.f) * (W / 2.f);
  return (x > W_2).select(
    x.constant(0.f),
    x.constant(scale) *
      (x.constant(beta) * (x.constant(1.f) - (x / x.constant(W_2))).sqrt()).bessel_i0());
}

template <int InPlane, int ThroughPlane>
auto Kernel<InPlane, ThroughPlane>::operator()(Point3 const r, float const scale) const -> KTensor
{
  KTensor k;
  if constexpr (ThroughPlane > 1) {
    constexpr Eigen::IndexList<FixIn, FixOne, FixOne> rshX;
    constexpr Eigen::IndexList<FixOne, FixIn, FixIn> brdX;
    constexpr Eigen::IndexList<FixOne, FixIn, FixOne> rshY;
    constexpr Eigen::IndexList<FixIn, FixOne, FixIn> brdY;
    constexpr Eigen::IndexList<FixOne, FixOne, FixIn> rshZ;
    constexpr Eigen::IndexList<FixIn, FixIn, FixOne> brdZ;
    auto const kx = (indices_.constant(r[0]) - indices_).square().reshape(rshX).broadcast(brdX);
    auto const ky = (indices_.constant(r[1]) - indices_).square().reshape(rshY).broadcast(brdY);
    auto const kz = (indices_.constant(r[2]) - indices_).square().reshape(rshZ).broadcast(brdZ);
    k = kx + ky + kz;
  } else {
    constexpr Eigen::IndexList<FixIn, FixOne, FixOne> rshX;
    constexpr Eigen::IndexList<FixOne, FixIn, FixOne> brdX;
    constexpr Eigen::IndexList<FixOne, FixIn, FixOne> rshY;
    constexpr Eigen::IndexList<FixIn, FixOne, FixOne> brdY;
    auto const kx = (indices_.constant(r[0]) - indices_).square().reshape(rshX).broadcast(brdX);
    auto const ky = (indices_.constant(r[1]) - indices_).square().reshape(rshY).broadcast(brdY);
    k = kx + ky;
  }
  k = KB<InPlane>(k, beta_, scale * kScale_);
  if (sqrt_) {
    // This is the worst possible way to do this but I cannot figure out what IFFT(SQRT(FFT(KB))) is
    Cx3 temp(InPlane, InPlane, ThroughPlane);
    temp = k.template cast<Cx>();
    fft_.reverse(temp);
    temp.sqrt();
    fft_.forward(temp);
    k = temp.real();
  }
  return k;
}

template struct Kernel<3, 3>;
template struct Kernel<3, 1>;
