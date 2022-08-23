#include "kernel-kb.hpp"

namespace rl {

template <int IP, int TP>
KaiserBessel<IP, TP>::KaiserBessel(float os)
  : beta_{(float)M_PI * sqrtf(pow(IP * (os - 0.5f) / os, 2.f) - 0.8f)}
{
  // Get the normalization factor
  scale_ = 1.f;
  scale_ = 1.f / Sum(k(Point3::Zero()));
  Log::Print(FMT_STRING("Kaiser-Bessel kernel <{},{}> β={} scale={} "), IP, TP, beta_, scale_);
}

template <int IP, int TP>
auto KaiserBessel<IP, TP>::k(Point3 const p) const -> KTensor
{
  auto const z2 = this->distSq(p);
  return (z2.sqrt() < 1.f).select(
    z2.constant(scale_) * (z2.constant(beta_) * (z2.constant(1.f) - z2).sqrt()).bessel_i0(), z2.constant(0.f));
}

template struct KaiserBessel<3, 1>;
template struct KaiserBessel<3, 3>;
template struct KaiserBessel<5, 1>;
template struct KaiserBessel<5, 5>;
template struct KaiserBessel<7, 1>;
template struct KaiserBessel<7, 7>;

} // namespace rl
