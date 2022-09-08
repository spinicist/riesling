#pragma once

#include <cmath>

namespace rl {

/*
 *  Flat-Iron / Barnet "Exponential of a Semi-Circle" kernel
 */
template <size_t W>
struct ExpSemi
{
  static constexpr size_t Width = W;
  static constexpr size_t PadWidth = (((W + 1) / 2) * 2) + 1;
  float const beta;

  ExpSemi(float const osamp)
    : beta{(float)M_PI * 0.98f * W * (1.f - 0.5f / osamp)}
  {
  }

  template <typename T>
  inline auto operator()(T const &z) const
  {
    return (z > 1.f).select(
      z.constant(0.f), (z.constant(beta) * ((z.constant(1.f) - z.square()).sqrt() - z.constant(1.f))).exp());
  }
};

} // namespace rl
