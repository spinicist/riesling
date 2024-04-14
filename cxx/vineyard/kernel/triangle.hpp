#pragma once

#include <cmath>

namespace rl {

/*
 * Linear / Bi-linear / Tri-linear
 */
template <int W>
struct Triangle
{
  static constexpr int Width = W;
  static constexpr int PadWidth = W;

  Triangle(float const osamp) {}

  template <typename T>
  inline auto operator()(T const &z) const
  {
    return (z > 1.f).select(z.constant(0.f), z.constant(1.f) - z);
  }
};

} // namespace rl
