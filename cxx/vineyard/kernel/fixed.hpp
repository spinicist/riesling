#pragma once

#include "kernel.hpp"

namespace rl {

template <int ND, int W> struct KernelSizes
{
};

template <int W> struct KernelSizes<1, W>
{
  using Type = Eigen::Sizes<W>;
};

template <int W> struct KernelSizes<2, W>
{
  using Type = Eigen::Sizes<W, W>;
};

template <int W> struct KernelSizes<3, W>
{
  using Type = Eigen::Sizes<W, W, W>;
};

template <typename Scalar, int ND, int W> struct FixedKernel : Kernel<Scalar, ND>
{
  using OneD = Eigen::TensorFixedSize<float, Eigen::Sizes<W>>;
  using Tensor = Eigen::TensorFixedSize<float, typename KernelSizes<ND, W>::Type>;
  using Point = Eigen::Matrix<float, ND, 1>;

  OneD centers;

  FixedKernel()
  {
    Eigen::TensorFixedSize<float, Eigen::Sizes<W>> pos;
    for (int ii = 0; ii < W; ii++) {
      centers(ii) = ii + 0.5f - (W / 2.f);
    }
  }

  virtual auto paddedWidth() const -> int final { return W; };
  virtual auto at(Point const p) const -> Eigen::Tensor<float, ND> final { return this->operator()(p); }
  virtual auto operator()(Point const p) const -> Tensor = 0;
  void         spread(std::array<int16_t, ND> const   c,
                      Point const                     p,
                      Sz<ND> const                    minCorner,
                      Eigen::Tensor<Scalar, 1> const &b,
                      Eigen::Tensor<Scalar, 1> const &y,
                      Eigen::Tensor<Scalar, ND + 2>  &x) const final
  {
    auto const  k = this->operator()(p);
    Index const hW = W / 2;
    Index const nC = x.dimension(0);
    Index const nB = b.size();
    for (Index i1 = 0; i1 < W; i1++) {
      Index const ii1 = i1 + c[ND - 1] - hW - minCorner[ND - 1];
      if constexpr (ND == 1) {
        float const kval = k(i1);
        for (Index ib = 0; ib < nB; ib++) {
          Scalar const bval = kval * b(ib);
          for (Index ic = 0; ic < nC; ic++) {
            Scalar const yval = y(ic) * bval;
            x(ic, ib, ii1) = x(ic, ib, ii1) + yval;
          }
        }
      } else {
        for (Index i2 = 0; i2 < W; i2++) {
          Index const ii2 = i2 + c[ND - 2] - hW - minCorner[ND - 2];
          if constexpr (ND == 2) {
            float const kval = k(i2, i1);
            for (Index ib = 0; ib < nB; ib++) {
              Scalar const bval = kval * b(ib);
              for (Index ic = 0; ic < nC; ic++) {
                Scalar const yval = y(ic) * bval;
                x(ic, ib, ii2, ii1) = x(ic, ib, ii2, ii1) + yval;
              }
            }
          } else {
            for (Index i3 = 0; i3 < W; i3++) {
              Index const ii3 = i3 + c[ND - 3] - hW - minCorner[ND - 3];
              float const kval = k(i3, i2, i1);
              for (Index ib = 0; ib < nB; ib++) {
                Scalar const bval = kval * b(ib);
                for (Index ic = 0; ic < nC; ic++) {
                  Scalar const yval = y(ic) * bval;
                  x(ic, ib, ii3, ii2, ii1) = x(ic, ib, ii3, ii2, ii1) + yval;
                }
              }
            }
          }
        }
      }
    }
  }

  void gather(std::array<int16_t, ND> const                                c,
              Point const                                                  p,
              Sz<ND> const                                                 minCorner,
              Eigen::Tensor<Scalar, 1> const                              &b,
              Eigen::TensorMap<Eigen::Tensor<Scalar, ND + 2> const> const &x,
              Eigen::TensorMap<Eigen::Tensor<Scalar, 1>>                  &y) const final
  {
    auto const k = this->operator()(p);
    Sz<ND>     offset;
    for (int id = 0; id < ND; id++) {
      offset[id] = c[id] - minCorner[id] - (W - 1) / 2;
    }
    Index const nC = x.dimension(0);
    Index const nB = b.size();
    for (Index i1 = 0; i1 < W; i1++) {
      Index const ii1 = i1 + offset[ND - 1];
      if constexpr (ND == 1) {
        float const kval = k(i1);
        for (Index ib = 0; ib < nB; ib++) {
          Scalar const bval = kval * b(ib);
          for (Index ic = 0; ic < nC; ic++) {
            Scalar const xval = x(ic, ib, ii1) * bval;
            y(ic) = y(ic) + xval;
          }
        }
      } else {
        for (Index i2 = 0; i2 < W; i2++) {
          Index const ii2 = i2 + offset[ND - 2];
          if constexpr (ND == 2) {
            float const kval = k(i2, i1);
            for (Index ib = 0; ib < nB; ib++) {
              Scalar const bval = kval * b(ib);
              for (Index ic = 0; ic < nC; ic++) {
                Scalar const xval = x(ic, ib, ii2, ii1) * bval;
                y(ic) = y(ic) + xval;
              }
            }
          } else {
            for (Index i3 = 0; i3 < W; i3++) {
              Index const ii3 = i3 + offset[ND - 3];
              float const kval = k(i3, i2, i1);
              for (Index ib = 0; ib < nB; ib++) {
                Scalar const bval = kval * b(ib);
                for (Index ic = 0; ic < nC; ic++) {
                  Scalar const xval = x(ic, ib, ii3, ii2, ii1) * bval;
                  y(ic) = y(ic) + xval;
                }
              }
            }
          }
        }
      }
    }
  }
};

} // namespace rl
