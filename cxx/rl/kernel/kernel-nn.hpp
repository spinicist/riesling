#pragma once

#include "../log.hpp"
#include "../tensors.hpp"

namespace rl {

template <typename Scalar, int ND> struct NearestNeighbour final
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

  auto paddedWidth() const -> int { return 1; }

  auto operator()(Point const) const -> Eigen::Tensor<float, ND>
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
        x(c[0], ic, 0) += yval;
      } else if constexpr (ND == 2) {
        x(c[0], c[1], ic, 0) += yval;
      } else if constexpr (ND == 3) {
        x(c[0], c[1], c[2], ic, 0) += yval;
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
          x(c[0], ic, ib) += bval;
        } else if constexpr (ND == 2) {
          x(c[0], c[1], ic, ib) += bval;
        } else if constexpr (ND == 3) {
          x(c[0], c[1], c[2], ic, ib) += bval;
        }
      }
    }
  }

  void gather(Eigen::Array<int16_t, ND, 1> const   c,
              Point const                         &p,
              Eigen::Tensor<Scalar, ND + 2> const &x,
              Eigen::Tensor<Scalar, 1>            &y) const final
  {
    Index const nC = x.dimension(1);
    for (Index ic = 0; ic < nC; ic++) {
      if constexpr (ND == 1) {
        y(ic) += x(c[0], ic, 0);
      } else if constexpr (ND == 2) {
        y(ic) += x(c[0], c[1], ic, 0);
      } else if constexpr (ND == 3) {
        y(ic) += x(c[0], c[1], c[2], ic, 0);
      }
    }
  }

  void gather(Eigen::Array<int16_t, ND, 1> const   c,
              Point const                         &p,
              Eigen::Tensor<Scalar, 1> const      &b,
              Eigen::Tensor<Scalar, ND + 2> const &x,
              Eigen::Tensor<Scalar, 1>            &y) const final
  {
    Index const nC = x.dimension(1);
    Index const nB = b.size();
    for (Index ic = 0; ic < nC; ic++) {
      for (Index ib = 0; ib < nB; ib++) {
        Scalar const bval = b(ib);
        if constexpr (ND == 1) {
          y(ic) += x(c[0], ic, ib) * bval;
        } else if constexpr (ND == 2) {
          y(ic) += x(c[0], c[1], ic, ib) * bval;
        } else if constexpr (ND == 3) {
          y(ic) += x(c[0], c[1], c[2], ic, ib) * bval;
        }
      }
    }
  }
};

} // namespace rl
