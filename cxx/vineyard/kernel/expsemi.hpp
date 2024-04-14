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
  inline auto operator()(T const &z, float const β) const
  {
    return (z > 1.f).select(z.constant(0.f), (z.constant(β) * ((z.constant(1.f) - z.square()).sqrt() - z.constant(1.f))).exp());
  }
};

} // namespace rl
