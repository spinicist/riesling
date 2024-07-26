#pragma once

#include "fixed.hpp"
#include "tensors.hpp"

namespace rl {

template <typename Scalar, int ND, typename Func> struct Radial final : Kernel<Scalar, ND>
{
  static constexpr int   Width = Func::Width;
  static constexpr int   PadWidth = Func::PadWidth;
  static constexpr float HalfWidth = Width / 2.f;
  using OneD = Eigen::TensorFixedSize<float, Eigen::Sizes<Func::PadWidth>>;
  using Tensor = typename FixedKernel<float, ND, Func>::Tensor;
  using Point = typename FixedKernel<float, ND, Func>::Point;
  using Pos = typename FixedKernel<float, ND, Func>::OneD;

  Func  f;
  float β, scale;
  OneD  centers;

  Radial(float const osamp)
    : f(osamp)
    , scale{1.f}
  {
    static_assert(ND < 4);
    for (int ii = 0; ii < Func::PadWidth; ii++) {
      this->centers(ii) = ii + 0.5f - (Func::PadWidth / 2.f);
    }
    scale = 1. / Norm(FixedKernel<Scalar, ND, Func>::K(f, 1.f, Point::Zero()));
    Log::Print("Radial, scale {}", scale);
  }

  virtual auto paddedWidth() const -> int final { return Func::PadWidth; }

  inline auto operator()(Point const p) const -> Eigen::Tensor<float, ND> final
  {
    return FixedKernel<Scalar, ND, Func>::K(f, scale, p);
  }

  void spread(std::array<int16_t, ND> const   c,
              Point const                    &p,
              Eigen::Tensor<Scalar, 1> const &b,
              Eigen::Tensor<Scalar, 1> const &y,
              Eigen::Tensor<Scalar, ND + 2>  &x) const final
  {
    FixedKernel<Scalar, ND, Func>::Spread(f, scale, c, p, b, y, x);
  }

  void gather(std::array<int16_t, ND> const                                c,
              Point const                                                 &p,
              Eigen::Tensor<Scalar, 1> const                              &b,
              Eigen::TensorMap<Eigen::Tensor<Scalar, ND + 2> const> const &x,
              Eigen::TensorMap<Eigen::Tensor<Scalar, 1>>                  &y) const final
  {
    FixedKernel<Scalar, ND, Func>::Gather(f, scale, c, p, b, x, y);
  }
};
} // namespace rl
