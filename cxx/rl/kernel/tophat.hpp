#pragma once

#include "../log.hpp"
#include <cmath>

namespace rl {

/*
 *  Top-Hat / Nearest-Neighbour Kernel
 */
template <int W> struct TopHat
{
  static constexpr int Width = W;
  static constexpr int FullWidth = W;

  TopHat(float const)
  {
    Log::Print("Kernel", "Top-Hat Width {}", W);
  }

  inline auto operator()(float const) const -> float
  {
    return 1.f;
  }
};

} // namespace rl
