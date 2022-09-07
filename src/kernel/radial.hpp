#pragma once

#include "internal.hpp"
#include "tensorOps.h"

namespace rl {

template <size_t N, typename Func>
struct Radial
{
  static constexpr size_t NDim = N;
  static constexpr size_t Width = Func::Width;
  static constexpr size_t PadWidth = Func::PadWidth;
  static constexpr float HalfWidth = Width / 2.f;
  using Tensor = typename KernelTypes<NDim, PadWidth>::Tensor;
  using Point = typename KernelTypes<NDim, PadWidth>::Point;
  using Pos = typename KernelTypes<NDim, PadWidth>::OneD;

  Func f;
  float scale;

  Radial(float const osamp)
    : f{osamp}
    , scale{1.f}
  {
    static_assert(N < 4);
    scale = 1. / Norm((*this)(Point::Zero()));
  }

  auto operator()(Point const p) const -> Tensor
  {
    Tensor z;
    Pos const z1 = ((Centers<PadWidth>() - p(N - 1)) / HalfWidth).square();
    if constexpr (N == 1) {
      z = z1.sqrt();
    } else {
      Pos const z2 = ((Centers<PadWidth>() - p(N - 2)) / HalfWidth).square();
      if constexpr (N == 2) {
        z = z2.reshape(Sz2{PadWidth, 1}).broadcast(Sz2{1, PadWidth}) +
            z1.reshape(Sz2{1, PadWidth}).broadcast(Sz2{PadWidth, 1});
      } else {
        Pos const z3 = ((Centers<PadWidth>() - p(N - 3)) / HalfWidth).square();
        z = z3.reshape(Sz3{PadWidth, 1, 1}).broadcast(Sz3{1, PadWidth, PadWidth}) +
            z2.reshape(Sz3{1, PadWidth, 1}).broadcast(Sz3{PadWidth, 1, PadWidth}) +
            z1.reshape(Sz3{1, 1, PadWidth}).broadcast(Sz3{PadWidth, PadWidth, 1});
      }
    }
    return f(z.sqrt()) * z.constant(scale);
  }
};

} // namespace rl