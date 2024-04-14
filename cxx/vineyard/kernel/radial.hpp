#pragma once

#include "fixed.hpp"
#include "tensorOps.hpp"

namespace rl {

template <typename Scalar, int N, typename Func>
struct Radial final : FixedKernel<Scalar, N, Func::PadWidth>
{
  static constexpr int NDim = N;
  static constexpr int Width = Func::Width;
  static constexpr int PadWidth = Func::PadWidth;
  static constexpr float  HalfWidth = Width / 2.f;
  using Tensor = typename FixedKernel<Scalar, NDim, PadWidth>::Tensor;
  using Point = typename FixedKernel<Scalar, NDim, PadWidth>::Point;
  using Pos = typename FixedKernel<Scalar, NDim, PadWidth>::OneD;

  Func  f;
  float β, scale;

  Radial(float const osamp)
    : β{f.β(osamp)}
    , scale{1.f}
  {
    static_assert(N < 4);
    scale = 1. / Norm((*this)(Point::Zero()));
    Log::Print("Radial, scale {}", scale);
  }

  void setOversampling(float const osamp) {
    β = f.β(osamp);
    scale = 1. / Norm((*this)(Point::Zero()));
  }

  auto operator()(Point const p) const -> Tensor
  {
    Tensor    z;
    Pos const z1 = ((this->centers - p(N - 1)) / HalfWidth).square();
    if constexpr (N == 1) {
      z = z1;
    } else {
      Pos const z2 = ((this->centers - p(N - 2)) / HalfWidth).square();
      if constexpr (N == 2) {
        z = z2.reshape(Sz2{PadWidth, 1}).broadcast(Sz2{1, PadWidth}) + z1.reshape(Sz2{1, PadWidth}).broadcast(Sz2{PadWidth, 1});
      } else {
        Pos const z3 = ((this->centers - p(N - 3)) / HalfWidth).square();
        z = z3.reshape(Sz3{PadWidth, 1, 1}).broadcast(Sz3{1, PadWidth, PadWidth}) +
            z2.reshape(Sz3{1, PadWidth, 1}).broadcast(Sz3{PadWidth, 1, PadWidth}) +
            z1.reshape(Sz3{1, 1, PadWidth}).broadcast(Sz3{PadWidth, PadWidth, 1});
      }
    }
    return f(z.sqrt(), β) * z.constant(scale);
  }
};

} // namespace rl
