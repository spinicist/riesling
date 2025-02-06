#pragma once

#include "../types.hpp"

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

template <int W, int PW> inline auto Z(float const p) -> Eigen::Array<float, PW, 1>
{
  constexpr float            HW = W / 2.f;
  constexpr float            L = 0.5f - PW / 2.f;
  Eigen::Array<float, PW, 1> z;
  for (Index ii = 0; ii < PW; ii++) {
    z[ii] = ((ii + L) - p) / HW;
  }
  return z;
}

template <int W, int PW, typename Func> inline auto K(Func const &f, float const p) -> Eigen::Array<float, PW, 1>
{
  constexpr float            HW = W / 2.f;
  constexpr float            L = 0.5f - PW / 2.f;
  Eigen::Array<float, PW, 1> z = Z<W, PW>(p);
  Eigen::Array<float, PW, 1> k;
  for (Index ii = 0; ii < PW; ii++) {
    k[ii] = f(z[ii]);
  }
  return k;
}

template <typename Scalar, int ND, typename Func> struct FixedKernel
{
};

template <typename Scalar, typename Func> struct FixedKernel<Scalar, 1, Func>
{
  constexpr static int W = Func::Width;
  static constexpr int PW = Func::FullWidth;
  using Array = Eigen::Array<float, PW, 1>;
  using Tensor = Eigen::TensorFixedSize<float, typename KernelSizes<1, PW>::Type>;
  using Point = Eigen::Matrix<float, 1, 1>;

  static inline auto Kernel(Func const &f, float const scale, Point const &p) -> Tensor
  {
    Tensor      k;
    Array const k0 = K<W, PW, Func>(f, p[0]) * scale;
    for (Index i0 = 0; i0 < PW; i0++) {
      k(i0) = k0[i0];
    }
    return k;
  }

  static inline void Spread(Func const                        &f,
                            float const                        scale,
                            Eigen::Array<int16_t, 1, 1> const &c,
                            Point const                       &p,
                            Eigen::Tensor<Scalar, 1> const    &y,
                            Eigen::Tensor<Scalar, 3>          &x)
  {
    Index const nC = y.dimension(0);
    assert(x.dimension(1) == nC);
    Array const k0 = K<W, PW, Func>(f, p[0]) * scale;
    for (Index i0 = 0; i0 < PW; i0++) {
      Index const ii0 = i0 + c[0] - PW / 2;
      for (Index ic = 0; ic < nC; ic++) {
        Scalar const yval = y(ic) * k0[i0];
        x(ii0, ic, 0) += yval;
      }
    }
  }

  static inline void Spread(Func const                        &f,
                            float const                        scale,
                            Eigen::Array<int16_t, 1, 1> const &c,
                            Point const                       &p,
                            Eigen::Tensor<Scalar, 1> const    &b,
                            Eigen::Tensor<Scalar, 1> const    &y,
                            Eigen::Tensor<Scalar, 3>          &x)
  {
    Index const nC = y.dimension(0);
    Index const nB = b.dimension(0);
    assert(x.dimension(1) == nC);
    assert(x.dimension(2) == nB);
    Array const k0 = K<W, PW, Func>(f, p[0]) * scale;
    for (Index i0 = 0; i0 < PW; i0++) {
      Index const ii0 = i0 + c[0] - PW / 2;
      for (Index ic = 0; ic < nC; ic++) {
        Scalar const yval = y(ic) * k0[i0];
        for (Index ib = 0; ib < nB; ib++) {
          Scalar const bval = yval * b(ib);
          x(ii0, ic, ib) += bval;
        }
      }
    }
  }

  static inline void Gather(Func const                        &f,
                            float const                        scale,
                            Eigen::Array<int16_t, 1, 1> const &c,
                            Point const                       &p,
                            Eigen::Tensor<Scalar, 3> const    &x,
                            Eigen::Tensor<Scalar, 1>          &y)
  {
    Index const nC = y.dimension(0);
    assert(x.dimension(1) == nC);
    Array const k0 = K<W, PW, Func>(f, p[0]) * scale;
    for (Index i0 = 0; i0 < PW; i0++) {
      Index const ii0 = i0 + c[0] - (PW - 1) / 2;
      for (Index ic = 0; ic < nC; ic++) {
        y(ic) += x(ii0, ic, 0) * k0[i0];
      }
    }
  }

  static inline void Gather(Func const                        &f,
                            float const                        scale,
                            Eigen::Array<int16_t, 1, 1> const &c,
                            Point const                       &p,
                            Eigen::Tensor<Scalar, 1> const    &b,
                            Eigen::Tensor<Scalar, 3> const    &x,
                            Eigen::Tensor<Scalar, 1>          &y)
  {
    Index const nC = y.dimension(0);
    Index const nB = b.dimension(0);
    assert(x.dimension(1) == nC);
    assert(x.dimension(2) == nB);
    Array const k0 = K<W, PW, Func>(f, p[0]) * scale;
    for (Index i0 = 0; i0 < PW; i0++) {
      Index const ii0 = i0 + c[0] - (PW - 1) / 2;
      for (Index ic = 0; ic < nC; ic++) {
        for (Index ib = 0; ib < nB; ib++) {
          y(ic) += x(ii0, ic, ib) * b(ib) * k0[i0];
        }
      }
    }
  }
};

template <typename Scalar, typename Func> struct FixedKernel<Scalar, 2, Func>
{
  constexpr static int W = Func::Width;
  static constexpr int PW = (((W + 1) / 2) * 2) + 1;
  using Array = Eigen::Array<float, PW, 1>;
  using Tensor = Eigen::TensorFixedSize<float, typename KernelSizes<2, PW>::Type>;
  using Point = Eigen::Matrix<float, 2, 1>;

  static inline auto Kernel(Func const &f, float const scale, Point const &p) -> Tensor
  {
    Tensor      k;
    Array const k1 = K<W, PW, Func>(f, p[1]);
    Array const k0 = K<W, PW, Func>(f, p[0]) * scale;
    for (Index i1 = 0; i1 < PW; i1++) {
      for (Index i0 = 0; i0 < PW; i0++) {
        k(i0, i1) = k0[i0] * k1[i1];
      }
    }
    return k;
  }

  static inline void Spread(Func const                        &f,
                            float const                        scale,
                            Eigen::Array<int16_t, 2, 1> const &c,
                            Point const                       &p,
                            Eigen::Tensor<Scalar, 1> const    &y,
                            Eigen::Tensor<Scalar, 4>          &x)
  {
    Index const nC = y.dimension(0);
    assert(x.dimension(2) == nC);
    Array const k1 = K<W, PW, Func>(f, p[1]);
    Array const k0 = K<W, PW, Func>(f, p[0]) * scale;
    for (Index i1 = 0; i1 < PW; i1++) {
      Index const ii1 = i1 + c[1] - PW / 2;
      for (Index i0 = 0; i0 < PW; i0++) {
        Index const ii0 = i0 + c[0] - PW / 2;
        float const k01 = k0[i0] * k1[i1];
        for (Index ic = 0; ic < nC; ic++) {
          Scalar const yval = y(ic) * k01;
          x(ii0, ii1, ic, 0) += yval;
        }
      }
    }
  }

  static inline void Spread(Func const                        &f,
                            float const                        scale,
                            Eigen::Array<int16_t, 2, 1> const &c,
                            Point const                       &p,
                            Eigen::Tensor<Scalar, 1> const    &b,
                            Eigen::Tensor<Scalar, 1> const    &y,
                            Eigen::Tensor<Scalar, 4>          &x)
  {
    Index const nC = y.dimension(0);
    Index const nB = b.dimension(0);
    assert(x.dimension(2) == nC);
    assert(x.dimension(3) == nB);
    Array const k1 = K<W, PW, Func>(f, p[1]);
    Array const k0 = K<W, PW, Func>(f, p[0]) * scale;
    for (Index i1 = 0; i1 < PW; i1++) {
      Index const ii1 = i1 + c[1] - PW / 2;
      for (Index i0 = 0; i0 < PW; i0++) {
        Index const ii0 = i0 + c[0] - PW / 2;
        float const k01 = k0[i0] * k1[i1];
        for (Index ic = 0; ic < nC; ic++) {
          Scalar const yval = y(ic) * k01;
          for (Index ib = 0; ib < nB; ib++) {
            Scalar const bval = yval * b(ib);
            x(ii0, ii1, ic, ib) += bval;
          }
        }
      }
    }
  }

  static inline void Gather(Func const                        &f,
                            float const                        scale,
                            Eigen::Array<int16_t, 2, 1> const &c,
                            Point const                       &p,
                            Eigen::Tensor<Scalar, 4> const    &x,
                            Eigen::Tensor<Scalar, 1>          &y)
  {
    Index const nC = y.dimension(0);
    assert(x.dimension(2) == nC);
    Array const k1 = K<W, PW, Func>(f, p[1]);
    Array const k0 = K<W, PW, Func>(f, p[0]) * scale;
    for (Index i1 = 0; i1 < PW; i1++) {
      Index const ii1 = i1 + c[1] - (PW - 1) / 2;
      for (Index i0 = 0; i0 < PW; i0++) {
        Index const ii0 = i0 + c[0] - (PW - 1) / 2;
        float const k01 = k0[i0] * k1[i1];
        for (Index ic = 0; ic < nC; ic++) {
          y(ic) += x(ii0, ii1, ic, 0) * k01;
        }
      }
    }
  }

  static inline void Gather(Func const                        &f,
                            float const                        scale,
                            Eigen::Array<int16_t, 2, 1> const &c,
                            Point const                       &p,
                            Eigen::Tensor<Scalar, 1> const    &b,
                            Eigen::Tensor<Scalar, 4> const    &x,
                            Eigen::Tensor<Scalar, 1>          &y)
  {
    Index const nC = y.dimension(0);
    Index const nB = b.dimension(0);
    assert(x.dimension(2) == nC);
    assert(x.dimension(3) == nB);
    Array const k1 = K<W, PW, Func>(f, p[1]);
    Array const k0 = K<W, PW, Func>(f, p[0]) * scale;
    for (Index i1 = 0; i1 < PW; i1++) {
      Index const ii1 = i1 + c[1] - (PW - 1) / 2;
      for (Index i0 = 0; i0 < PW; i0++) {
        Index const ii0 = i0 + c[0] - (PW - 1) / 2;
        float const k01 = k0[i0] * k1[i1];
        for (Index ic = 0; ic < nC; ic++) {
          for (Index ib = 0; ib < nB; ib++) {
            y(ic) += x(ii0, ii1, ic, ib) * b(ib) * k01;
          }
        }
      }
    }
  }
};

template <typename Scalar, typename Func> struct FixedKernel<Scalar, 3, Func>
{
  constexpr static int W = Func::Width;
  static constexpr int PW = (((W + 1) / 2) * 2) + 1;
  using Array = Eigen::Array<float, PW, 1>;
  using Tensor = Eigen::TensorFixedSize<float, typename KernelSizes<3, PW>::Type>;
  using Point = Eigen::Matrix<float, 3, 1>;

  static inline auto Kernel(Func const &f, float const scale, Point const &p) -> Tensor
  {
    Tensor      k;
    Array const k2 = K<W, PW, Func>(f, p[2]);
    Array const k1 = K<W, PW, Func>(f, p[1]);
    Array const k0 = K<W, PW, Func>(f, p[0]) * scale;
    for (Index i2 = 0; i2 < PW; i2++) {
      for (Index i1 = 0; i1 < PW; i1++) {
        float const k12 = k1[i1] * k2[i2];
        for (Index i0 = 0; i0 < PW; i0++) {
          k(i0, i1, i2) = k0[i0] * k12;
        }
      }
    }
    return k;
  }

  static inline void Spread(Func const                        &f,
                            float const                        scale,
                            Eigen::Array<int16_t, 3, 1> const &c,
                            Point const                       &p,
                            Eigen::Tensor<Scalar, 1> const    &y,
                            Eigen::Tensor<Scalar, 5>          &x)
  {
    Index const nC = y.dimension(0);
    assert(x.dimension(3) == nC);
    Array const k2 = K<W, PW, Func>(f, p[2]);
    Array const k1 = K<W, PW, Func>(f, p[1]);
    Array const k0 = K<W, PW, Func>(f, p[0]) * scale;
    for (Index i2 = 0; i2 < PW; i2++) {
      Index const ii2 = i2 + c[2] - PW / 2;
      for (Index i1 = 0; i1 < PW; i1++) {
        Index const ii1 = i1 + c[1] - PW / 2;
        float const k12 = k1[i1] * k2[i2];
        for (Index i0 = 0; i0 < PW; i0++) {
          Index const ii0 = i0 + c[0] - PW / 2;
          float const k = k0[i0] * k12;
          for (Index ic = 0; ic < nC; ic++) {
            Scalar const yval = y(ic) * k;
            x(ii0, ii1, ii2, ic, 0) += yval;
          }
        }
      }
    }
  }

  static inline void Spread(Func const                        &f,
                            float const                        scale,
                            Eigen::Array<int16_t, 3, 1> const &c,
                            Point const                       &p,
                            Eigen::Tensor<Scalar, 1> const    &b,
                            Eigen::Tensor<Scalar, 1> const    &y,
                            Eigen::Tensor<Scalar, 5>          &x)
  {
    Index const nC = y.dimension(0);
    Index const nB = b.dimension(0);
    assert(x.dimension(3) == nC);
    assert(x.dimension(4) == nB);
    Array const k2 = K<W, PW, Func>(f, p[2]);
    Array const k1 = K<W, PW, Func>(f, p[1]);
    Array const k0 = K<W, PW, Func>(f, p[0]) * scale;
    for (Index ib = 0; ib < nB; ib++) {
      Scalar const bb = b(ib);
      for (Index ic = 0; ic < nC; ic++) {
        Scalar const cb = y(ic) * bb;
        for (Index i2 = 0; i2 < PW; i2++) {
          Index const  ii2 = i2 + c[2] - PW / 2;
          Scalar const k2cb = k2[i2] * cb;
          for (Index i1 = 0; i1 < PW; i1++) {
            Index const  ii1 = i1 + c[1] - PW / 2;
            Scalar const k12cb = k1[i1] * k2cb;
            for (Index i0 = 0; i0 < PW; i0++) {
              Index const  ii0 = i0 + c[0] - PW / 2;
              Scalar const k012cb = k0[i0] * k12cb;
              x(ii0, ii1, ii2, ic, ib) += k012cb;
            }
          }
        }
      }
    }
  }

  static inline void Gather(Func const                        &f,
                            float const                        scale,
                            Eigen::Array<int16_t, 3, 1> const &c,
                            Point const                       &p,
                            Eigen::Tensor<Scalar, 5> const    &x,
                            Eigen::Tensor<Scalar, 1>          &y)
  {
    Index const nC = y.dimension(0);
    assert(x.dimension(3) == nC);
    Array const k2 = K<W, PW, Func>(f, p[2]);
    Array const k1 = K<W, PW, Func>(f, p[1]);
    Array const k0 = K<W, PW, Func>(f, p[0]) * scale;
    for (Index i2 = 0; i2 < PW; i2++) {
      Index const ii2 = i2 + c[2] - (PW - 1) / 2;
      for (Index i1 = 0; i1 < PW; i1++) {
        Index const ii1 = i1 + c[1] - (PW - 1) / 2;
        float const k12 = k1[i1] * k2[i2];
        for (Index i0 = 0; i0 < PW; i0++) {
          Index const ii0 = i0 + c[0] - (PW - 1) / 2;
          float const k012 = k0[i0] * k12;
          for (Index ic = 0; ic < nC; ic++) {
            y(ic) += x(ii0, ii1, ii2, ic, 0) * k012;
          }
        }
      }
    }
  }

  static inline void Gather(Func const                        &f,
                            float const                        scale,
                            Eigen::Array<int16_t, 3, 1> const &c,
                            Point const                       &p,
                            Eigen::Tensor<Scalar, 1> const    &b,
                            Eigen::Tensor<Scalar, 5> const    &x,
                            Eigen::Tensor<Scalar, 1>          &y)
  {
    Index const nC = y.dimension(0);
    Index const nB = b.dimension(0);
    assert(x.dimension(3) == nC);
    assert(x.dimension(4) == nB);
    Array const k2 = K<W, PW, Func>(f, p[2]);
    Array const k1 = K<W, PW, Func>(f, p[1]);
    Array const k0 = K<W, PW, Func>(f, p[0]) * scale;
    for (Index ib = 0; ib < nB; ib++) {
      for (Index i2 = 0; i2 < PW; i2++) {
        Index const  ii2 = i2 + c[2] - (PW - 1) / 2;
        Scalar const k2b = k2(i2) * b(ib);
        for (Index i1 = 0; i1 < PW; i1++) {
          Index const ii1 = i1 + c[1] - (PW - 1) / 2;
          float const k12b = k1(i1) * k12b;
          for (Index i0 = 0; i0 < PW; i0++) {
            Index const ii0 = i0 + c[0] - (PW - 1) / 2;
            float const k012b = k0(i0) * k12b;
            for (Index ic = 0; ic < nC; ic++) {
              y(ic) += x(ii0, ii1, ii2, ic, ib) * k012b;
            }
          }
        }
      }
    }
  }
};

} // namespace rl
