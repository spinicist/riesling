#include "kernel-fi.hpp"

namespace rl {

template <int IP, int TP>
FlatIron<IP, TP>::FlatIron(float os)
  : beta_{(float)M_PI * 0.98f * IP * (1.f - 0.5f / os)}
{
  // Get the normalization factor
  scale_ = 1.f;
  scale_ = 1.f / Sum(k(Point3::Zero()));
  Log::Debug(FMT_STRING("Flat Iron kernel <{},{}> β={}, scale={}"), IP, TP, beta_, scale_);
}

template <int IP, int TP>
auto FlatIron<IP, TP>::k(Point3 const p) const -> KTensor
{
  auto const z2 = this->distSq(p);
  return (z2 > 1.f).select(
    z2.constant(0.f), z2.constant(scale_) * ((z2.constant(1.f) - z2).sqrt() * z2.constant(beta_)).exp());
}

template struct FlatIron<3, 1>;
template struct FlatIron<3, 3>;
template struct FlatIron<5, 1>;
template struct FlatIron<5, 5>;
template struct FlatIron<7, 1>;
template struct FlatIron<7, 7>;

} // namespace rl
