#pragma once

#include "kernel-fixed.hpp"

#include "expsemi.hpp"
#include "kaiser.hpp"

#include "../tensors.hpp"

#include <fmt/ranges.h>

namespace rl {

template <typename Scalar, int ND, typename Func> struct Kernel final
{
  static constexpr int Width = Func::Width;
  static constexpr int PadWidth = (((Width + 1) / 2) * 2) + 1;
  using Tensor = typename FixedKernel<float, ND, Func>::Tensor;
  using Point = typename FixedKernel<float, ND, Func>::Point;

  Func  f;
  float scale;

  Kernel(float const osamp)
    : f(osamp)
    , scale{1.f}
  {
    static_assert(ND < 4);
    scale = 1. / Norm<false>(FixedKernel<Scalar, ND, Func>::Kernel(f, 1.f, Point::Zero()));
    Log::Print("Kernel", "Width {} Scale {}", Func::Width, scale);
  }

  virtual auto paddedWidth() const -> int final { return PadWidth; }

  inline auto operator()(Point const p = Point::Zero()) const -> Eigen::Tensor<float, ND>
  {
    Tensor                   k = FixedKernel<Scalar, ND, Func>::Kernel(f, scale, p);
    Eigen::Tensor<float, ND> k2 = k;
    return k;
  }

  void spread(Eigen::Array<int16_t, ND, 1> const c,
              Point const                       &p,
              Eigen::Tensor<Scalar, 1> const    &y,
              Eigen::Tensor<Scalar, ND + 2>     &x) const
  {
    FixedKernel<Scalar, ND, Func>::Spread(f, scale, c, p, y, x);
  }

  void spread(Eigen::Array<int16_t, ND, 1> const c,
              Point const                       &p,
              Eigen::Tensor<Scalar, 1> const    &b,
              Eigen::Tensor<Scalar, 1> const    &y,
              Eigen::Tensor<Scalar, ND + 2>     &x) const
  {
    FixedKernel<Scalar, ND, Func>::Spread(f, scale, c, p, b, y, x);
  }

  void gather(Eigen::Array<int16_t, ND, 1> const   c,
              Point const                         &p,
              Eigen::Tensor<Scalar, ND + 2> const &x,
              Eigen::Tensor<Scalar, 1>            &y) const
  {
    FixedKernel<Scalar, ND, Func>::Gather(f, scale, c, p, x, y);
  }

  void gather(Eigen::Array<int16_t, ND, 1> const   c,
              Point const                         &p,
              Eigen::Tensor<Scalar, 1> const      &b,
              Eigen::Tensor<Scalar, ND + 2> const &x,
              Eigen::Tensor<Scalar, 1>            &y) const
  {
    FixedKernel<Scalar, ND, Func>::Gather(f, scale, c, p, b, x, y);
  }
};

} // namespace rl