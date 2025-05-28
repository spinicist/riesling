#pragma once

#include "expsemi.hpp"
#include "tophat.hpp"

#include "../tensors.hpp"

namespace rl {

template <int ND, int W> struct FixedSize
{
};

template <int W> struct FixedSize<1, W>
{
  using T = Eigen::Sizes<W>;
};

template <int W> struct FixedSize<2, W>
{
  using T = Eigen::Sizes<W, W>;
};

template <int W> struct FixedSize<3, W>
{
  using T = Eigen::Sizes<W, W, W>;
};

template <typename T, int ND, int SZ> using FixedTensor = Eigen::TensorFixedSize<T, typename FixedSize<ND, SZ>::T>;

namespace {
template <typename Func> inline auto K(Func const f, float const p) -> FixedTensor<float, 1, Func::FullWidth>
{
  constexpr float HW = Func::Width / 2.f;
  constexpr float L = 0.5f - Func::FullWidth / 2.f;

  FixedTensor<float, 1, Func::FullWidth> k;
  for (Index ii = 0; ii < Func::FullWidth; ii++) {
    k[ii] = f(((ii + L) - p) / HW);
  }
  return k;
}

template <typename Func> inline auto KS(Func const f, float const p, float const s) -> FixedTensor<float, 1, Func::FullWidth>
{
  constexpr float HW = Func::Width / 2.f;
  constexpr float L = 0.5f - Func::FullWidth / 2.f;

  FixedTensor<float, 1, Func::FullWidth> k;
  for (Index ii = 0; ii < Func::FullWidth; ii++) {
    k[ii] = f(((ii + L) - p) / HW) * s;
  }
  return k;
}

} // namespace

template <int ND, typename Func> struct Kernel
{
  static constexpr int Width = Func::Width;
  static constexpr int FullWidth = Func::FullWidth;

  using Array = Eigen::Array<float, FullWidth, 1>;
  using Tensor = FixedTensor<float, ND, Func::FullWidth>;
  using Point = Eigen::Matrix<float, ND, 1>;

  Func  f;
  float scale;

  Kernel(float const osamp)
    : f(osamp)
    , scale{1.f}
  {
    static_assert(ND < 4);
    scale = 1. / Norm<false>(this->operator()());
    Log::Print("Kernel", "Width {} Scale {}", Func::Width, scale);
  }

  inline auto operator()(Point const p = Point::Zero()) const -> Tensor
  {
    auto const k0 = KS<Func>(f, p[0], scale);
    if constexpr (ND == 1) {
      return k0;
    } else {
      Tensor     k;
      auto const k1 = K<Func>(f, p[1]);
      if constexpr (ND == 2) {
        for (Index i1 = 0; i1 < Func::FullWidth; i1++) {
          for (Index i0 = 0; i0 < Func::FullWidth; i0++) {
            k(i0, i1) = k0(i0) * k1(i1);
          }
        }
      } else if constexpr (ND == 3) {
        auto const k2 = K<Func>(f, p[2]);
        for (Index i2 = 0; i2 < Func::FullWidth; i2++) {
          for (Index i1 = 0; i1 < Func::FullWidth; i1++) {
            float const k12 = k1(i1) * k2(i2);
            for (Index i0 = 0; i0 < Func::FullWidth; i0++) {
              k(i0, i1, i2) = k0(i0) * k12;
            }
          }
        }
      }
      return k;
    }
  }
};

} // namespace rl