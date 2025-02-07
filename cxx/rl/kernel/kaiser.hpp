#pragma once

#include "../log.hpp"
#include <cmath>

namespace rl {

template <int W>
struct KaiserBessel
{
  static constexpr int Width = W;
  static constexpr int FullWidth = (((W + 1) / 2) * 2) + 1;

  float β;

  // Use Fessler's approximate formula to avoid problems with apodization
  KaiserBessel(float const osamp) : β{(float)M_PI * 2.34f * Width * osamp / 2.f}
{ Log::Print("Kernel", "Kaiser-Bessel width {}", W); }

  inline auto operator()(float const z2) const
  {
    return z2 > 1.f ? 0.f : β * Eigen::numext::bessel_i0(std::sqrt(1.f - z2));
  }
};

} // namespace rl
