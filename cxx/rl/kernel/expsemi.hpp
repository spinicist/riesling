#pragma once

#include "../log/log.hpp"
#include <cmath>

namespace rl {

/*
 *  Flat-Iron / Barnet "Exponential of a Semi-Circle" kernel
 */
template <int W> struct ExpSemi
{
  static constexpr int Width = W;
  static constexpr int FullWidth = (((W + 1) / 2) * 2) + 1;
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

// template <int W> struct ExpSemiSinc
// {
//   static constexpr int Width = W;
//   static constexpr int FullWidth = (((W + 1) / 2) * 2) + 1;
//   float β, os;

//   ExpSemiSinc(float const osamp)
//     : β{(float)M_PI * 0.98f * W * (1.f - 0.5f / osamp)}, os{osamp}
//   {
//     Log::Print("Kernel", "Exponential Semi-Circle Sinc width {} β {} over-sampling {}", W, β, os);
//   }

//   inline auto operator()(float const z) const -> float
//   {
//     float const piz = z * os * (Width / 2.f) * (float)M_PI;
//     float const sincz = z == 0.f ? 1.f : std::sin(piz) / piz;
//     float const z2 = z*z;
//     // fmt::print(stderr, "z {} piz {} sincz {} z2 {}\n", z, piz, sincz, z2);
//     return z2 > 1.f ? 0.f : std::exp(β * std::sqrt(1.f - z2) - 1.f) * sincz;
//   }
// };

} // namespace rl
