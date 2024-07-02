#pragma once

#include "fixed.hpp"
#include "tensors.hpp"

namespace rl {

template <typename Scalar, int N, typename Func> struct Rectilinear final : FixedKernel<Scalar, N, Func::PadWidth>
{
  static constexpr int   NDim = N;
  static constexpr int   Width = Func::Width;
  static constexpr int   PadWidth = Func::PadWidth;
  static constexpr float HalfWidth = Width / 2.f;
  using Tensor = typename FixedKernel<Scalar, NDim, PadWidth>::Tensor;
  using Point = typename FixedKernel<Scalar, NDim, PadWidth>::Point;
  using Pos = typename FixedKernel<Scalar, NDim, PadWidth>::OneD;

  Func  f;
  float β, scale;

  Rectilinear(float const osamp)
    : β{f.β(osamp)}
    , scale{1.f}
  {
    static_assert(N < 4);
    scale = 1. / Norm((*this)(Point::Zero()));
    Log::Print("Rectilinear, scale {}", scale);
  }

  auto operator()(Point const p) const -> Tensor
  {
    Pos const k1 = f((this->centers - p(N - 1)).abs() / HalfWidth, β) * scale;
    if constexpr (N == 1) {
      return k1;
    } else {
      Pos const k2 = f((this->centers - p(N - 2)).abs() / HalfWidth, β);
      if constexpr (N == 2) {
        return k2.reshape(Sz2{PadWidth, 1}).broadcast(Sz2{1, PadWidth}) *
               k1.reshape(Sz2{1, PadWidth}).broadcast(Sz2{PadWidth, 1});
      } else {
        Pos const k3 = f((this->centers - p(N - 3)).abs() / HalfWidth, β);
        return k3.reshape(Sz3{PadWidth, 1, 1}).broadcast(Sz3{1, PadWidth, PadWidth}) *
               k2.reshape(Sz3{1, PadWidth, 1}).broadcast(Sz3{PadWidth, 1, PadWidth}) *
               k1.reshape(Sz3{1, 1, PadWidth}).broadcast(Sz3{PadWidth, PadWidth, 1});
      }
    }
  }
};

} // namespace rl
