#pragma once

#include "fixed.hpp"

namespace rl {

template <typename Scalar, int ND> struct NearestNeighbour final : Kernel<Scalar, ND>
{
  static constexpr int   Width = 1;
  static constexpr int   PadWidth = 1;
  static constexpr float HalfWidth = 1;
  using Tensor = typename FixedKernel<float, ND, PadWidth>::Tensor;
  using Point = typename FixedKernel<float, ND, PadWidth>::Point;
  using Pos = typename FixedKernel<float, ND, PadWidth>::OneD;

  NearestNeighbour()
  {
    static_assert(ND < 4);
    Log::Print("Nearest-neighbour FixedKernel");
  }

  auto paddedWidth() const -> int final { return 1; }

  auto operator()(Point const) const -> Eigen::Tensor<float, ND> final
  {
    Tensor z;
    z.setConstant(1.f);
    return z;
  }

  void spread(std::array<int16_t, ND> const   c,
              Point const                     p,
              Eigen::Tensor<Scalar, 1> const &b,
              Eigen::Tensor<Scalar, 1> const &y,
              Eigen::Tensor<Scalar, ND + 2>  &x) const final
  {
    Index const nC = x.dimension(0);
    Index const nB = b.size();
    for (Index ib = 0; ib < nB; ib++) {
      Scalar const bval = b(ib);
      for (Index ic = 0; ic < nC; ic++) {
        Scalar const yval = y(ic) * bval;
        if constexpr (ND == 1) {
          x(ic, ib, c[0]) = x(ic, ib, c[0]) + yval;
        } else if constexpr (ND == 2) {
          x(ic, ib, c[0], c[1]) = x(ic, ib, c[0], c[1]) + yval;
        } else if constexpr (ND == 3) {
          x(ic, ib, c[0], c[1], c[2]) = x(ic, ib, c[0], c[1], c[2]) + yval;
        }
      }
    }
  }

  void gather(std::array<int16_t, ND> const                                c,
              Point const                                                  p,
              Eigen::Tensor<Scalar, 1> const                              &b,
              Eigen::TensorMap<Eigen::Tensor<Scalar, ND + 2> const> const &x,
              Eigen::TensorMap<Eigen::Tensor<Scalar, 1>>                  &y) const final
  {
    Index const nC = x.dimension(0);
    Index const nB = b.size();
    for (Index ib = 0; ib < nB; ib++) {
      Scalar const bval = b(ib);
      for (Index ic = 0; ic < nC; ic++) {
        if constexpr (ND == 1) {
          y(ic) += x(ic, ib, c[0]) * bval;
        } else if constexpr (ND == 2) {
          y(ic) += x(ic, ib, c[0], c[1]) * bval;
        } else if constexpr (ND == 3) {
          y(ic) += x(ic, ib, c[0], c[1], c[2]) * bval;
        }
      }
    }
  }
};

} // namespace rl
