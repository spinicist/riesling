#pragma once

#include "fixed.hpp"

namespace rl {

template <typename Scalar, int N>
struct NearestNeighbour final : FixedKernel<Scalar, N, 1>
{
  static constexpr int NDim = N;
  static constexpr int Width = 1;
  static constexpr int PadWidth = 1;
  static constexpr float  HalfWidth = 1;
  using Tensor = typename FixedKernel<Scalar, NDim, PadWidth>::Tensor;
  using Point = typename FixedKernel<Scalar, NDim, PadWidth>::Point;
  using Pos = typename FixedKernel<Scalar, NDim, PadWidth>::OneD;

  NearestNeighbour()
  {
    static_assert(N < 4);
    Log::Print("Nearest-neighbour FixedKernel");
  }

  auto operator()(Point const ) const -> Tensor
  {
    Tensor z;
    z.setConstant(1.f);
    return z;
  }

  void setOversampling(float const) {}
};

} // namespace rl
