#pragma once

#include <cmath>

namespace rl {

/*
 * Linear / Bi-linear / Tri-linear
 */
template <int W> struct Triangle
{
  static constexpr int Width = W;
  static constexpr int FullWidth = W;

  Triangle(float const) {}

  template <typename T> inline auto operator()(T const &z) const
  {
    return (z > 1.f).select(z.constant(0.f), z.constant(1.f) - z);
  }
};

} // namespace rl
