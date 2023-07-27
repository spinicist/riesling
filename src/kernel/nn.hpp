#pragma once

#include "fixed.hpp"

namespace rl {

template <typename Scalar, size_t N>
struct NearestNeighbour final : FixedKernel<Scalar, N, 1>
{
  static constexpr size_t NDim = N;
  static constexpr size_t Width = 1;
  static constexpr size_t PadWidth = 1;
  static constexpr float  HalfWidth = 1;
  using Tensor = typename FixedKernel<Scalar, NDim, PadWidth>::Tensor;
  using Point = typename FixedKernel<Scalar, NDim, PadWidth>::Point;
  using Pos = typename FixedKernel<Scalar, NDim, PadWidth>::OneD;

  NearestNeighbour()
  {
    static_assert(N < 4);
    Log::Print("Nearest-neighbour FixedKernel");
  }

  auto operator()(Point const p) const -> Tensor
  {
    Tensor z;
    z.setConstant(1.f);
    return z;
  }
};

} // namespace rl
