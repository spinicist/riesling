#pragma once

#include "internal.hpp"
#include "tensorOps.h"

namespace rl {

template <size_t N>
struct NearestNeighbour
{
  static constexpr size_t NDim = N;
  static constexpr size_t Width = 1;
  static constexpr size_t PadWidth = 1;
  static constexpr float HalfWidth = 1;
  using Tensor = typename KernelTypes<NDim, PadWidth>::Tensor;
  using Point = typename KernelTypes<NDim, PadWidth>::Point;
  using Pos = typename KernelTypes<NDim, PadWidth>::OneD;

  NearestNeighbour(float const)
  {
    static_assert(N < 4);
  }

  auto operator()(Point const p) const -> Tensor
  {
    Tensor z;
    z.setConstant(1.f);
    return z;
  }
};

} // namespace rl
