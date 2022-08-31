#pragma once

#include <cmath>

namespace rl {

/*
 * Linear / Bi-linear / Tri-linear
 */
template <size_t W>
struct Triangle
{
  static constexpr size_t Width = W;
  static constexpr size_t PadWidth = W;

  Triangle(float const osamp)
  {
    // Log::Debug("Flat-Iron Kernel W={} β={}", W, beta);
  }

  template <typename T>
  inline auto operator()(T const &z) const
  {
    return (z > 1.f).select(z.constant(0.f), z.constant(1.f) - z);
  };
};

} // namespace rl
