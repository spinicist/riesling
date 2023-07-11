#pragma once

#include "kernel.hpp"
#include "tensorOps.hpp"

namespace rl {

template <size_t N, typename Func>
struct Rectilinear final : Kernel<N, Func::PadWidth>
{
  static constexpr size_t NDim = N;
  static constexpr size_t Width = Func::Width;
  static constexpr size_t PadWidth = Func::PadWidth;
  static constexpr float  HalfWidth = Width / 2.f;
  using Tensor = typename Kernel<NDim, PadWidth>::Tensor;
  using Point = typename Kernel<NDim, PadWidth>::Point;
  using Pos = typename Kernel<NDim, PadWidth>::OneD;

  Func  f;
  float scale;

  Rectilinear(float const osamp)
    : f{osamp}
    , scale{1.f}
  {
    static_assert(N < 4);
    scale = 1. / Norm((*this)(Point::Zero()));
    Log::Print("Rectilinear, scale {}", scale);
  }

  auto operator()(Point const p) const -> Tensor
  {
    Pos const k1 = f((this->centers - p(N - 1)).abs() / HalfWidth) * scale;
    if constexpr (N == 1) {
      return k1;
    } else {
      Pos const k2 = f((this->centers - p(N - 2)).abs() / HalfWidth);
      if constexpr (N == 2) {
        return k2.reshape(Sz2{PadWidth, 1}).broadcast(Sz2{1, PadWidth}) *
               k1.reshape(Sz2{1, PadWidth}).broadcast(Sz2{PadWidth, 1});
      } else {
        Pos const k3 = f((this->centers - p(N - 3)).abs() / HalfWidth);
        return k3.reshape(Sz3{PadWidth, 1, 1}).broadcast(Sz3{1, PadWidth, PadWidth}) *
               k2.reshape(Sz3{1, PadWidth, 1}).broadcast(Sz3{PadWidth, 1, PadWidth}) *
               k1.reshape(Sz3{1, 1, PadWidth}).broadcast(Sz3{PadWidth, PadWidth, 1});
      }
    }
  }
};

} // namespace rl
