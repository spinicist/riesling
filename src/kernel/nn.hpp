#pragma once

#include "kernel.hpp"

namespace rl {

template <size_t N>
struct NearestNeighbour final : Kernel<N, 1>
{
  static constexpr size_t NDim = N;
  static constexpr size_t Width = 1;
  static constexpr size_t PadWidth = 1;
  static constexpr float  HalfWidth = 1;
  using Tensor = typename Kernel<NDim, PadWidth>::Tensor;
  using Point = typename Kernel<NDim, PadWidth>::Point;
  using Pos = typename Kernel<NDim, PadWidth>::OneD;

  NearestNeighbour(float const)
  {
    static_assert(N < 4);
    Log::Print("Nearest-neighbour kernel");
  }

  auto operator()(Point const p) const -> Tensor
  {
    Tensor z;
    z.setConstant(1.f);
    return z;
  }
};

} // namespace rl
