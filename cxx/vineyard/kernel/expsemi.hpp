#pragma once

#include "log.hpp"
#include <cmath>

namespace rl {

/*
 *  Flat-Iron / Barnet "Exponential of a Semi-Circle" kernel
 */
template <int W>
struct ExpSemi
{
  static constexpr int Width = W;
  static constexpr int PadWidth = (((W + 1) / 2) * 2) + 1;

  static float β(float const osamp) { return (float)M_PI * 0.98f * W * (1.f - 0.5f / osamp); }

  ExpSemi() { Log::Print("Exponential Semi-Circle kernel width {}", W); }

  template <typename T>
  inline auto operator()(T const &z2, float const β) const
  {
    return (z2 > 1.f).select(z2.constant(0.f), (z2.constant(β) * ((z2.constant(1.f) - z2).sqrt() - z2.constant(1.f))).exp());
  }
};

} // namespace rl
