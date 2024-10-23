#pragma once

#include "log.hpp"
#include <cmath>

namespace rl {

/*
 *  Flat-Iron / Barnet "Exponential of a Semi-Circle" kernel
 */
template <int W> struct ExpSemi
{
  static constexpr int Width = W;
  float β;

  ExpSemi(float const osamp)
    : β{(float)M_PI * 0.98f * W * (1.f - 0.5f / osamp)}
  {
    Log::Print("Kernel", "Exponential Semi-Circle width {} β {}", W, β);
  }

  inline auto operator()(float const z) const -> float
  {
    float const z2 = z*z;
    return z2 > 1.f ? 0.f : std::exp(β * std::sqrt(1.f - z2) - 1.f);
  }
};

} // namespace rl
