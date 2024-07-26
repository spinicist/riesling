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

template <typename Scalar, int ND, typename Func> struct FixedKernel
{
};

template <typename Scalar, typename Func> struct FixedKernel<Scalar, 1, Func>
{
  constexpr static int W = Func::PadWidth;
  using OneD = Eigen::TensorFixedSize<float, Eigen::Sizes<W>>;
  using Tensor = Eigen::TensorFixedSize<float, typename KernelSizes<1, W>::Type>;
  using Point = Eigen::Matrix<float, 1, 1>;

  static inline void Spread(std::array<int16_t, 1> const   &corner,
                            Tensor const                   &k,
                            Eigen::Tensor<Scalar, 1> const &b,
                            Eigen::Tensor<Scalar, 1> const &y,
                            Eigen::Tensor<Scalar, 3>       &x)
  {
    Index const nC = x.dimension(0);
    Index const nB = b.size();
    for (Index i0 = 0; i0 < W; i0++) {
      Index const ii0 = i0 + corner[0] - W / 2;
      float const kval = k(i0);
      for (Index ib = 0; ib < nB; ib++) {
        Scalar const bval = kval * b(ib);
        for (Index ic = 0; ic < nC; ic++) {
          Scalar const yval = y(ic) * bval;
          x(ic, ib, ii0) += yval;
        }
      }
    }
  }

  static inline void Gather(std::array<int16_t, 1> const                           &corner,
                            Tensor const                                           &k,
                            Eigen::Tensor<Scalar, 1> const                         &b,
                            Eigen::TensorMap<Eigen::Tensor<Scalar, 3> const> const &x,
                            Eigen::TensorMap<Eigen::Tensor<Scalar, 1>>             &y)
  {
    Index const nC = x.dimension(0);
    Index const nB = b.size();
    for (Index i0 = 0; i0 < W; i0++) {
      Index const ii0 = i0 + corner[0] - (W - 1) / 2;
      for (Index ib = 0; ib < nB; ib++) {
        Scalar const bval = b(ib);
        for (Index ic = 0; ic < nC; ic++) {
          y(ic) += x(ic, ib, ii0) * k(i0) * bval;
        }
      }
    }
  }
};

template <typename Scalar, typename Func> struct FixedKernel<Scalar, 2, Func>
{
  constexpr static int W = Func::PadWidth;
  using OneD = Eigen::TensorFixedSize<float, Eigen::Sizes<W>>;
  using Tensor = Eigen::TensorFixedSize<float, typename KernelSizes<2, W>::Type>;
  using Point = Eigen::Matrix<float, 2, 1>;

  static inline void Spread(std::array<int16_t, 2> const   &c,
                            Tensor const                   &k,
                            Eigen::Tensor<Scalar, 1> const &b,
                            Eigen::Tensor<Scalar, 1> const &y,
                            Eigen::Tensor<Scalar, 4>       &x)
  {
    Index const nC = x.dimension(0);
    Index const nB = b.size();
    for (Index i1 = 0; i1 < W; i1++) {
      Index const ii1 = i1 + c[1] - W / 2;
      for (Index i0 = 0; i0 < W; i0++) {
        Index const ii0 = i0 + c[0] - W / 2;
        float const kval = k(i0, i1);
        for (Index ib = 0; ib < nB; ib++) {
          Scalar const bval = kval * b(ib);
          for (Index ic = 0; ic < nC; ic++) {
            Scalar const yval = y(ic) * bval;
            x(ic, ib, ii0, ii1) += yval;
          }
        }
      }
    }
  }

  static inline void Gather(std::array<int16_t, 2> const                           &c,
                            Tensor const                                           &k,
                            Eigen::Tensor<Scalar, 1> const                         &b,
                            Eigen::TensorMap<Eigen::Tensor<Scalar, 4> const> const &x,
                            Eigen::TensorMap<Eigen::Tensor<Scalar, 1>>             &y)
  {
    Index const nC = x.dimension(0);
    Index const nB = b.size();
    for (Index i1 = 0; i1 < W; i1++) {
      Index const ii1 = i1 + c[1] - (W - 1) / 2;
      for (Index i0 = 0; i0 < W; i0++) {
        Index const ii0 = i0 + c[0] - (W - 1) / 2;
        for (Index ib = 0; ib < nB; ib++) {
          Scalar const bval = b(ib);
          for (Index ic = 0; ic < nC; ic++) {
            y(ic) += x(ic, ib, ii0, ii1) * k(i0, i1) * bval;
          }
        }
      }
    }
  }
};

template <typename Scalar, typename Func> struct FixedKernel<Scalar, 3, Func>
{
  constexpr static int W = Func::PadWidth;
  using OneD = Eigen::TensorFixedSize<float, Eigen::Sizes<W>>;
  using Tensor = Eigen::TensorFixedSize<float, typename KernelSizes<3, W>::Type>;
  using Point = Eigen::Matrix<float, 3, 1>;

  static inline void Spread(std::array<int16_t, 3> const    c,
                            Tensor const                    k,
                            Eigen::Tensor<Scalar, 1> const &b,
                            Eigen::Tensor<Scalar, 1> const &y,
                            Eigen::Tensor<Scalar, 5>       &x)
  {
    Index const nC = x.dimension(0);
    Index const nB = b.dimension(0);
    for (Index i2 = 0; i2 < W; i2++) {
      Index const ii2 = i2 + c[2] - W / 2;
      for (Index i1 = 0; i1 < W; i1++) {
        Index const ii1 = i1 + c[1] - W / 2;
        for (Index i0 = 0; i0 < W; i0++) {
          Index const ii0 = i0 + c[0] - W / 2;
          float const kval = k(i0, i1, i2);
          for (Index ic = 0; ic < nC; ic++) {
            for (Index ib = 0; ib < nB; ib++) {
              Scalar const bval = kval * b(ib);
              Scalar const yval = y(ic) * bval;
              x(ib, ic, ii0, ii1, ii2) += yval;
            }
          }
        }
      }
    }
  }

  static inline void Spread2(float const                     β,
                             std::array<int16_t, 3> const   &c,
                             Point const                    &p,
                             Eigen::Tensor<Scalar, 1> const &b,
                             Eigen::Tensor<Scalar, 1> const &y,
                             Eigen::Tensor<Scalar, 5>       &x)
  {
    Index const nC = x.dimension(1);
    Index const nB = b.dimension(0);
    if (nB == 1) {
      for (Index i2 = 0; i2 < W; i2++) {
        Index const ii2 = i2 + c[2] - W / 2;
        float const z2 = pow(((i2 + (0.5f - W / 2.f)) - p[2]) / (W / 2.f), 2);
        for (Index i1 = 0; i1 < W; i1++) {
          Index const ii1 = i1 + c[1] - W / 2;
          float const z1 = pow(((i1 + (0.5f - W / 2.f)) - p[1]) / (W / 2.f), 2);
          for (Index i0 = 0; i0 < W; i0++) {
            Index const ii0 = i0 + c[0] - W / 2;
            float const z0 = pow(((i0 + (0.5f - W / 2.f)) - p[0]) / (W / 2.f), 2);
            float const kval = Func::Scalar(z0 + z1 + z2, β);
            for (Index ic = 0; ic < nC; ic++) {
              Scalar const yval = y(ic) * kval;
              x(0, ic, ii0, ii1, ii2) += yval;
            }
          }
        }
      }
    } else {
      for (Index i2 = 0; i2 < W; i2++) {
        Index const ii2 = i2 + c[2] - W / 2;
        float const z2 = pow(((i2 + (0.5f - W / 2.f)) - p[2]) / (W / 2.f), 2);
        for (Index i1 = 0; i1 < W; i1++) {
          Index const ii1 = i1 + c[1] - W / 2;
          float const z1 = pow(((i1 + (0.5f - W / 2.f)) - p[1]) / (W / 2.f), 2);
          for (Index i0 = 0; i0 < W; i0++) {
            Index const ii0 = i0 + c[0] - W / 2;
            float const z0 = pow(((i0 + (0.5f - W / 2.f)) - p[0]) / (W / 2.f), 2);
            float const kval = Func::Scalar(z0 + z1 + z2, β);
            for (Index ic = 0; ic < nC; ic++) {
              Scalar const yval = y(ic) * kval;
              for (Index ib = 0; ib < nB; ib++) {
                Scalar const bval = yval * b(ib);
                x(ib, ic, ii0, ii1, ii2) += bval;
              }
            }
          }
        }
      }
    }
  }

  static inline void Gather(std::array<int16_t, 3> const                            c,
                            Tensor const                                            k,
                            Eigen::Tensor<Scalar, 1> const                         &b,
                            Eigen::TensorMap<Eigen::Tensor<Scalar, 5> const> const &x,
                            Eigen::TensorMap<Eigen::Tensor<Scalar, 1>>             &y)
  {
    Index const nC = x.dimension(1);
    Index const nB = b.dimension(0);
    for (Index i2 = 0; i2 < W; i2++) {
      Index const ii2 = i2 + c[2] - (W - 1) / 2;
      for (Index i1 = 0; i1 < W; i1++) {
        Index const ii1 = i1 + c[1] - (W - 1) / 2;
        for (Index i0 = 0; i0 < W; i0++) {
          Index const ii0 = i0 + c[0] - (W - 1) / 2;
          for (Index ic = 0; ic < nC; ic++) {
            for (Index ib = 0; ib < nB; ib++) {
              Scalar const bval = b(ib);
              y(ic) += x(ib, ic, ii0, ii1, ii2) * k(i0, i1, i2) * bval;
            }
          }
        }
      }
    }
  }

  static inline void Gather2(float const                                             β,
                             std::array<int16_t, 3> const                           &c,
                             Point const                                            &p,
                             Eigen::Tensor<Scalar, 1> const                         &b,
                             Eigen::TensorMap<Eigen::Tensor<Scalar, 5> const> const &x,
                             Eigen::TensorMap<Eigen::Tensor<Scalar, 1>>             &y)
  {
    Index const nC = x.dimension(1);
    Index const nB = b.dimension(0);
    if (nB == 1) {
      for (Index i2 = 0; i2 < W; i2++) {
        Index const ii2 = i2 + c[2] - (W - 1) / 2;
        float const z2 = pow(((i2 + (0.5f - W / 2.f)) - p[2]) / (W / 2.f), 2);
        for (Index i1 = 0; i1 < W; i1++) {
          Index const ii1 = i1 + c[1] - (W - 1) / 2;
          float const z1 = pow(((i1 + (0.5f - W / 2.f)) - p[1]) / (W / 2.f), 2);
          for (Index i0 = 0; i0 < W; i0++) {
            Index const ii0 = i0 + c[0] - (W - 1) / 2;
            float const z0 = pow(((i0 + (0.5f - W / 2.f)) - p[0]) / (W / 2.f), 2);
            float const kval = Func::Scalar(z0 + z1 + z2, β);
            for (Index ic = 0; ic < nC; ic++) {
              y(ic) += x(0, ic, ii0, ii1, ii2) * kval;
            }
          }
        }
      }
    } else {
      for (Index i2 = 0; i2 < W; i2++) {
        Index const ii2 = i2 + c[2] - (W - 1) / 2;
        float const z2 = pow(((i2 + (0.5f - W / 2.f)) - p[2]) / (W / 2.f), 2);
        for (Index i1 = 0; i1 < W; i1++) {
          Index const ii1 = i1 + c[1] - (W - 1) / 2;
          float const z1 = pow(((i1 + (0.5f - W / 2.f)) - p[1]) / (W / 2.f), 2);
          for (Index i0 = 0; i0 < W; i0++) {
            Index const ii0 = i0 + c[0] - (W - 1) / 2;
            float const z0 = pow(((i0 + (0.5f - W / 2.f)) - p[0]) / (W / 2.f), 2);
            float const kval = Func::Scalar(z0 + z1 + z2, β);
            for (Index ic = 0; ic < nC; ic++) {
              for (Index ib = 0; ib < nB; ib++) {
                y(ic) += x(ib, ic, ii0, ii1, ii2) * b(ib) * kval;
              }
            }
          }
        }
      }
    }
  }
};

} // namespace rl
