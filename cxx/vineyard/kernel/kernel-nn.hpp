#pragma once

#include "kernel.hpp"

#include "log.hpp"

namespace rl {

template <typename Scalar, int ND> struct NearestNeighbour final : KernelBase<Scalar, ND>
{
  static constexpr int Width = 1;
  static constexpr int PadWidth = 1;
  using Tensor = Eigen::TensorFixedSize<float, typename KernelSizes<ND, 1>::Type>;
  using Point = Eigen::Matrix<float, ND, 1>;
  using Pos = Eigen::TensorFixedSize<float, typename KernelSizes<ND, 1>::Type>;

  NearestNeighbour()
  {
    static_assert(ND < 4);
    Log::Print("Kernel", "Nearest-neighbour");
  }

  auto paddedWidth() const -> int final { return 1; }

  auto operator()(Point const) const -> Eigen::Tensor<float, ND> final
  {
    Tensor z;
    z.setConstant(1.f);
    return z;
  }

  void spread(Eigen::Array<int16_t, ND, 1> const c,
              Point const                       &p,
              Eigen::Tensor<Scalar, 1> const    &y,
              Eigen::Tensor<Scalar, ND + 2>     &x) const final
  {
    Index const nC = x.dimension(1);
    for (Index ic = 0; ic < nC; ic++) {
      Scalar const yval = y(ic);
      if constexpr (ND == 1) {
        x(0, ic, c[0]) += yval;
      } else if constexpr (ND == 2) {
        x(0, ic, c[0], c[1]) += yval;
      } else if constexpr (ND == 3) {
        x(0, ic, c[0], c[1], c[2]) += yval;
      }
    }
  }

  void spread(Eigen::Array<int16_t, ND, 1> const c,
              Point const                       &p,
              Eigen::Tensor<Scalar, 1> const    &b,
              Eigen::Tensor<Scalar, 1> const    &y,
              Eigen::Tensor<Scalar, ND + 2>     &x) const final
  {
    Index const nC = x.dimension(1);
    Index const nB = b.size();
    for (Index ic = 0; ic < nC; ic++) {
      Scalar const yval = y(ic);
      for (Index ib = 0; ib < nB; ib++) {
        Scalar const bval = b(ib) * yval;
        if constexpr (ND == 1) {
          x(ib, ic, c[0]) += bval;
        } else if constexpr (ND == 2) {
          x(ib, ic, c[0], c[1]) += bval;
        } else if constexpr (ND == 3) {
          x(ib, ic, c[0], c[1], c[2]) += bval;
        }
      }
    }
  }

  void gather(Eigen::Array<int16_t, ND, 1> const                           c,
              Point const                                                 &p,
              Eigen::TensorMap<Eigen::Tensor<Scalar, ND + 2> const> const &x,
              Eigen::TensorMap<Eigen::Tensor<Scalar, 1>>                  &y) const final
  {
    Index const nC = x.dimension(1);
    for (Index ic = 0; ic < nC; ic++) {
      if constexpr (ND == 1) {
        y(ic) += x(0, ic, c[0]);
      } else if constexpr (ND == 2) {
        y(ic) += x(0, ic, c[0], c[1]);
      } else if constexpr (ND == 3) {
        y(ic) += x(0, ic, c[0], c[1], c[2]);
      }
    }
  }

  void gather(Eigen::Array<int16_t, ND, 1> const                           c,
              Point const                                                 &p,
              Eigen::Tensor<Scalar, 1> const                              &b,
              Eigen::TensorMap<Eigen::Tensor<Scalar, ND + 2> const> const &x,
              Eigen::TensorMap<Eigen::Tensor<Scalar, 1>>                  &y) const final
  {
    Index const nC = x.dimension(1);
    Index const nB = b.size();
    for (Index ic = 0; ic < nC; ic++) {
      for (Index ib = 0; ib < nB; ib++) {
        Scalar const bval = b(ib);
        if constexpr (ND == 1) {
          y(ic) += x(ib, ic, c[0]) * bval;
        } else if constexpr (ND == 2) {
          y(ic) += x(ib, ic, c[0], c[1]) * bval;
        } else if constexpr (ND == 3) {
          y(ic) += x(ib, ic, c[0], c[1], c[2]) * bval;
        }
      }
    }
  }
};

} // namespace rl
