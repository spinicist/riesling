#pragma once

#include <complex>
#include <fmt/std.h>
#include <fmt/ranges.h>
#include <fmt/ostream.h>

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
  constexpr static int   W = Func::PadWidth;
  constexpr static float HW = Func::Width / 2.f;
  constexpr static float L = (0.5f - W / 2.f);
  using OneD = Eigen::TensorFixedSize<float, Eigen::Sizes<W>>;
  using Tensor = Eigen::TensorFixedSize<float, typename KernelSizes<1, W>::Type>;
  using Point = Eigen::Matrix<float, 1, 1>;

  static inline auto K(Func const &f, float const scale, Point const &p) -> Tensor
  {
    Tensor k;
    for (Index i0 = 0; i0 < W; i0++) {
      float const z0 = pow(((i0 + L) - p[0]) / HW, 2);
      k(i0) = f(z0) * scale;
    }
    return k;
  }

  static inline void Spread(Func const                     &f,
                            float const                     scale,
                            std::array<int16_t, 1> const   &c,
                            Point const                    &p,
                            Eigen::Tensor<Scalar, 1> const &y,
                            Eigen::Tensor<Scalar, 3>       &x)
  {
    Index const nC = x.dimension(1);
    for (Index i0 = 0; i0 < W; i0++) {
      Index const ii0 = i0 + c[0] - W / 2;
      float const z0 = pow(((i0 + L) - p[0]) / HW, 2);
      float const kval = f(z0) * scale;
      for (Index ic = 0; ic < nC; ic++) {
        Scalar const yval = y(ic) * kval;
        x(0, ic, ii0) += yval;
      }
    }
  }

  static inline void Spread(Func const                     &f,
                            float const                     scale,
                            std::array<int16_t, 1> const   &c,
                            Point const                    &p,
                            Eigen::Tensor<Scalar, 1> const &b,
                            Eigen::Tensor<Scalar, 1> const &y,
                            Eigen::Tensor<Scalar, 3>       &x)
  {
    Index const nC = x.dimension(1);
    Index const nB = b.dimension(0);
    for (Index i0 = 0; i0 < W; i0++) {
      Index const ii0 = i0 + c[0] - W / 2;
      float const z0 = pow(((i0 + L) - p[0]) / HW, 2);
      float const kval = f(z0) * scale;
      for (Index ic = 0; ic < nC; ic++) {
        Scalar const yval = y(ic) * kval;
        for (Index ib = 0; ib < nB; ib++) {
          Scalar const bval = yval * b(ib);
          x(ib, ic, ii0) += bval;
        }
      }
    }
  }

  static inline void Gather(Func const                                             &f,
                            float const                                             scale,
                            std::array<int16_t, 1> const                           &c,
                            Point const                                            &p,
                            Eigen::TensorMap<Eigen::Tensor<Scalar, 3> const> const &x,
                            Eigen::TensorMap<Eigen::Tensor<Scalar, 1>>             &y)
  {
    Index const nC = x.dimension(1);
    for (Index i0 = 0; i0 < W; i0++) {
      Index const ii0 = i0 + c[0] - (W - 1) / 2;
      float const z0 = pow(((i0 + L) - p[0]) / HW, 2);
      float const kval = f(z0) * scale;
      for (Index ic = 0; ic < nC; ic++) {
        y(ic) += x(0, ic, ii0) * kval;
      }
    }
  }

  static inline void Gather(Func const                                             &f,
                            float const                                             scale,
                            std::array<int16_t, 1> const                           &c,
                            Point const                                            &p,
                            Eigen::Tensor<Scalar, 1> const                         &b,
                            Eigen::TensorMap<Eigen::Tensor<Scalar, 3> const> const &x,
                            Eigen::TensorMap<Eigen::Tensor<Scalar, 1>>             &y)
  {
    Index const nC = x.dimension(1);
    Index const nB = b.dimension(0);
    for (Index i0 = 0; i0 < W; i0++) {
      Index const ii0 = i0 + c[0] - (W - 1) / 2;
      float const z0 = pow(((i0 + L) - p[0]) / HW, 2);
      float const kval = f(z0) * scale;
      for (Index ic = 0; ic < nC; ic++) {
        for (Index ib = 0; ib < nB; ib++) {
          y(ic) += x(ib, ic, ii0) * b(ib) * kval;
        }
      }
    }
  }
};

template <typename Scalar, typename Func> struct FixedKernel<Scalar, 2, Func>
{
  constexpr static int   W = Func::PadWidth;
  constexpr static float HW = Func::Width / 2.f;
  constexpr static float L = (0.5f - W / 2.f);
  using OneD = Eigen::TensorFixedSize<float, Eigen::Sizes<W>>;
  using Tensor = Eigen::TensorFixedSize<float, typename KernelSizes<2, W>::Type>;
  using Point = Eigen::Matrix<float, 2, 1>;

  static inline auto K(Func const &f, float const scale, Point const &p) -> Tensor
  {
    Tensor k;
    for (Index i1 = 0; i1 < W; i1++) {
      float const z1 = pow(((i1 + L) - p[1]) / HW, 2);
      for (Index i0 = 0; i0 < W; i0++) {
        float const z0 = pow(((i0 + L) - p[0]) / HW, 2);
        k(i0, i1) = f(z0 + z1) * scale;
      }
    }
    return k;
  }

  static inline void Spread(Func const                     &f,
                            float const                     scale,
                            std::array<int16_t, 2> const   &c,
                            Point const                    &p,
                            Eigen::Tensor<Scalar, 1> const &y,
                            Eigen::Tensor<Scalar, 4>       &x)
  {
    Index const nC = x.dimension(1);
    for (Index i1 = 0; i1 < W; i1++) {
      Index const ii1 = i1 + c[1] - W / 2;
      float const z1 = pow(((i1 + L) - p[1]) / HW, 2);
      for (Index i0 = 0; i0 < W; i0++) {
        Index const ii0 = i0 + c[0] - W / 2;
        float const z0 = pow(((i0 + L) - p[0]) / HW, 2);
        float const kval = f(z0 + z1) * scale;
        for (Index ic = 0; ic < nC; ic++) {
          Scalar const yval = y(ic) * kval;
          x(0, ic, ii0, ii1) += yval;
        }
      }
    }
  }

  static inline void Spread(Func const                     &f,
                            float const                     scale,
                            std::array<int16_t, 2> const   &c,
                            Point const                    &p,
                            Eigen::Tensor<Scalar, 1> const &b,
                            Eigen::Tensor<Scalar, 1> const &y,
                            Eigen::Tensor<Scalar, 4>       &x)
  {
    Index const nC = x.dimension(1);
    Index const nB = b.dimension(0);
    for (Index i1 = 0; i1 < W; i1++) {
      Index const ii1 = i1 + c[1] - W / 2;
      float const z1 = pow(((i1 + L) - p[1]) / HW, 2);
      for (Index i0 = 0; i0 < W; i0++) {
        Index const ii0 = i0 + c[0] - W / 2;
        float const z0 = pow(((i0 + L) - p[0]) / HW, 2);
        float const kval = f(z0 + z1) * scale;
        for (Index ic = 0; ic < nC; ic++) {
          Scalar const yval = y(ic) * kval;
          for (Index ib = 0; ib < nB; ib++) {
            Scalar const bval = yval * b(ib);
            x(ib, ic, ii0, ii1) += bval;
          }
        }
      }
    }
  }

  static inline void Gather(Func const                                             &f,
                            float const                                             scale,
                            std::array<int16_t, 2> const                           &c,
                            Point const                                            &p,
                            Eigen::TensorMap<Eigen::Tensor<Scalar, 4> const> const &x,
                            Eigen::TensorMap<Eigen::Tensor<Scalar, 1>>             &y)
  {
    Index const nC = x.dimension(1);
    for (Index i1 = 0; i1 < W; i1++) {
      Index const ii1 = i1 + c[1] - (W - 1) / 2;
      float const z1 = pow(((i1 + L) - p[1]) / HW, 2);
      for (Index i0 = 0; i0 < W; i0++) {
        Index const ii0 = i0 + c[0] - (W - 1) / 2;
        float const z0 = pow(((i0 + L) - p[0]) / HW, 2);
        float const kval = f(z0 + z1) * scale;
        for (Index ic = 0; ic < nC; ic++) {
          y(ic) += x(0, ic, ii0, ii1) * kval;
        }
      }
    }
  }

  static inline void Gather(Func const                                             &f,
                            float const                                             scale,
                            std::array<int16_t, 2> const                           &c,
                            Point const                                            &p,
                            Eigen::Tensor<Scalar, 1> const                         &b,
                            Eigen::TensorMap<Eigen::Tensor<Scalar, 4> const> const &x,
                            Eigen::TensorMap<Eigen::Tensor<Scalar, 1>>             &y)
  {
    Index const nC = x.dimension(1);
    Index const nB = b.dimension(0);
    for (Index i1 = 0; i1 < W; i1++) {
      Index const ii1 = i1 + c[1] - (W - 1) / 2;
      float const z1 = pow(((i1 + L) - p[1]) / HW, 2);
      for (Index i0 = 0; i0 < W; i0++) {
        Index const ii0 = i0 + c[0] - (W - 1) / 2;
        float const z0 = pow(((i0 + L) - p[0]) / HW, 2);
        float const kval = f(z0 + z1) * scale;
        for (Index ic = 0; ic < nC; ic++) {
          for (Index ib = 0; ib < nB; ib++) {
            y(ic) += x(ib, ic, ii0, ii1) * b(ib) * kval;
          }
        }
      }
    }
  }
};

template <typename Scalar, typename Func> struct FixedKernel<Scalar, 3, Func>
{
  constexpr static int   W = Func::PadWidth;
  constexpr static float HW = Func::Width / 2.f;
  constexpr static float L = (0.5f - W / 2.f);
  using OneD = Eigen::TensorFixedSize<float, Eigen::Sizes<W>>;
  using Tensor = Eigen::TensorFixedSize<float, typename KernelSizes<3, W>::Type>;
  using Point = Eigen::Matrix<float, 3, 1>;

  static inline auto K(Func const &f, float const scale, Point const &p) -> Tensor
  {
    Tensor k;
    for (Index i2 = 0; i2 < W; i2++) {
      float const z2 = pow(((i2 + L) - p[2]) / HW, 2);
      for (Index i1 = 0; i1 < W; i1++) {
        float const z1 = pow(((i1 + L) - p[1]) / HW, 2);
        for (Index i0 = 0; i0 < W; i0++) {
          float const z0 = pow(((i0 + L) - p[0]) / HW, 2);
          k(i0, i1, i2) = f(z0 + z1 + z2) * scale;
        }
      }
    }
    return k;
  }

  static inline void Spread(Func const                     &f,
                            float const                     scale,
                            std::array<int16_t, 3> const   &c,
                            Point const                    &p,
                            Eigen::Tensor<Scalar, 1> const &y,
                            Eigen::Tensor<Scalar, 5>       &x)
  {
    Index const nC = x.dimension(1);
    for (Index i2 = 0; i2 < W; i2++) {
      Index const ii2 = i2 + c[2] - W / 2;
      float const z2 = pow(((i2 + L) - p[2]) / HW, 2);
      for (Index i1 = 0; i1 < W; i1++) {
        Index const ii1 = i1 + c[1] - W / 2;
        float const z1 = pow(((i1 + L) - p[1]) / HW, 2);
        for (Index i0 = 0; i0 < W; i0++) {
          Index const ii0 = i0 + c[0] - W / 2;
          float const z0 = pow(((i0 + L) - p[0]) / HW, 2);
          float const kval = f(z0 + z1 + z2) * scale;
          for (Index ic = 0; ic < nC; ic++) {
            Scalar const yval = y(ic) * kval;
            x(0, ic, ii0, ii1, ii2) += yval;
          }
        }
      }
    }
  }

  static inline void Spread(Func const                     &f,
                            float const                     scale,
                            std::array<int16_t, 3> const   &c,
                            Point const                    &p,
                            Eigen::Tensor<Scalar, 1> const &b,
                            Eigen::Tensor<Scalar, 1> const &y,
                            Eigen::Tensor<Scalar, 5>       &x)
  {
    Index const nC = x.dimension(1);
    Index const nB = b.dimension(0);
    for (Index i2 = 0; i2 < W; i2++) {
      Index const ii2 = i2 + c[2] - W / 2;
      float const z2 = pow(((i2 + L) - p[2]) / HW, 2);
      for (Index i1 = 0; i1 < W; i1++) {
        Index const ii1 = i1 + c[1] - W / 2;
        float const z1 = pow(((i1 + L) - p[1]) / HW, 2);
        for (Index i0 = 0; i0 < W; i0++) {
          Index const ii0 = i0 + c[0] - W / 2;
          float const z0 = pow(((i0 + L) - p[0]) / HW, 2);
          float const kval = f(z0 + z1 + z2) * scale;
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

  static inline void Gather(Func const                                             &f,
                            float const                                             scale,
                            std::array<int16_t, 3> const                           &c,
                            Point const                                            &p,
                            Eigen::TensorMap<Eigen::Tensor<Scalar, 5> const> const &x,
                            Eigen::TensorMap<Eigen::Tensor<Scalar, 1>>             &y)
  {
    Index const nC = x.dimension(1);
    for (Index i2 = 0; i2 < W; i2++) {
      Index const ii2 = i2 + c[2] - (W - 1) / 2;
      float const z2 = pow(((i2 + L) - p[2]) / HW, 2);
      for (Index i1 = 0; i1 < W; i1++) {
        Index const ii1 = i1 + c[1] - (W - 1) / 2;
        float const z1 = pow(((i1 + L) - p[1]) / HW, 2);
        for (Index i0 = 0; i0 < W; i0++) {
          Index const ii0 = i0 + c[0] - (W - 1) / 2;
          float const z0 = pow(((i0 + L) - p[0]) / HW, 2);
          float const kval = f(z0 + z1 + z2) * scale;
          for (Index ic = 0; ic < nC; ic++) {
            y(ic) += x(0, ic, ii0, ii1, ii2) * kval;
          }
        }
      }
    }
  }

  static inline void Gather(Func const                                             &f,
                            float const                                             scale,
                            std::array<int16_t, 3> const                           &c,
                            Point const                                            &p,
                            Eigen::Tensor<Scalar, 1> const                         &b,
                            Eigen::TensorMap<Eigen::Tensor<Scalar, 5> const> const &x,
                            Eigen::TensorMap<Eigen::Tensor<Scalar, 1>>             &y)
  {
    Index const nC = x.dimension(1);
    Index const nB = b.dimension(0);
    for (Index i2 = 0; i2 < W; i2++) {
      Index const ii2 = i2 + c[2] - (W - 1) / 2;
      float const z2 = pow(((i2 + L) - p[2]) / HW, 2);
      for (Index i1 = 0; i1 < W; i1++) {
        Index const ii1 = i1 + c[1] - (W - 1) / 2;
        float const z1 = pow(((i1 + L) - p[1]) / HW, 2);
        for (Index i0 = 0; i0 < W; i0++) {
          Index const ii0 = i0 + c[0] - (W - 1) / 2;
          float const z0 = pow(((i0 + L) - p[0]) / HW, 2);
          float const kval = f(z0 + z1 + z2) * scale;
          for (Index ic = 0; ic < nC; ic++) {
            for (Index ib = 0; ib < nB; ib++) {
              y(ic) += x(ib, ic, ii0, ii1, ii2) * b(ib) * kval;
            }
          }
        }
      }
    }
  }
};

} // namespace rl
