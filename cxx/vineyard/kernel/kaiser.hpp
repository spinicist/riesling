#pragma once

#include "log.hpp"
#include <cmath>

namespace rl {

template <int W>
struct KaiserBessel
{
  static constexpr int Width = W;
  static constexpr int PadWidth = (((W + 1) / 2) * 2) + 1;

  static float β(float const osamp) { return (float)M_PI * 2.34f * Width * osamp / 2.f; }

  // Use Fessler's approximate formula to avoid problems with apodization
  KaiserBessel() { Log::Print("Kaiser-Bessel kernel width {}", W); }

  template <typename T>
  inline auto operator()(T const &z2, float const β) const
  {
    return (z2 > 1.f).select(z2.constant(0.f), (z2.constant(β) * (z2.constant(1.f) - z2)).sqrt().bessel_i0());
  }
};

} // namespace rl
