#pragma once

#include "log.hpp"
#include <cmath>

namespace rl {

template <size_t W>
struct KaiserBessel
{
  static constexpr size_t Width = W;
  static constexpr size_t PadWidth = (((W + 1) / 2) * 2) + 1;

  static float β(float const osamp) { return (float)M_PI * 2.34f * Width * osamp / 2.f; }

  // Use Fessler's approximate formula to avoid problems with apodization
  KaiserBessel() { Log::Print("Kaiser-Bessel kernel width {}", W); }

  template <typename T>
  inline auto operator()(T const &z, float const β) const
  {
    return (z > 1.f).select(z.constant(0.f), (z.constant(β) * (z.constant(1.f) - z.square())).sqrt().bessel_i0());
  }
};

} // namespace rl
