#pragma once

#include "tensorOps.h"
#include "types.h"

#include <utility>

namespace rl {

struct NewKernel
{
  virtual ~NewKernel(){};
};

template <size_t N, size_t W>
struct NewSizedKernel : NewKernel
{
  static const Index NDim = N;
  static const Index Width = W;
  // Welcome to C++
  using KTensor = decltype([]<std::size_t... Is>(std::index_sequence<Is...>) {
    return std::declval<Eigen::TensorFixedSize<float, Eigen::Sizes<(Is, W + 2)...>>>();
  }(std::make_index_sequence<N>()));
  using KPoint = Eigen::Matrix<float, N, 1>;

  auto distSq(KPoint const p) const -> KTensor;
  virtual KTensor k(KPoint const offset) const = 0;
};

template <size_t N, size_t W>
struct NewFlatIron final : NewSizedKernel<N, W>
{
  using KTensor = typename NewSizedKernel<N, W>::KTensor;
  using KPoint = typename NewSizedKernel<N, W>::KPoint;
  static Index const PadWidth = W + 2;
  float beta, scale;

  NewFlatIron(float const os)
    : beta{(float)M_PI * 0.98f * W * (1.f - 0.5f / os)}
    , scale{1.f}
  {
    scale = 1. / Norm(k(KPoint::Zero()));
  }

  virtual auto k(KPoint const p) const -> KTensor
  {
    KTensor k;
    if constexpr (N == 2) {
      for (size_t iy = 0; iy < PadWidth; iy++) {
        float const py = std::abs(iy - p(1) - (PadWidth / 2)) / (W / 2.f);
        if (py > 1.f) {
          for (size_t ix = 0; ix < PadWidth; ix++) {
            k(ix, iy) = 0.f;
          }
        } else {
          float const ky = std::exp(beta * (std::sqrt(1.f - py * py) - 1.f));
          for (size_t ix = 0; ix < PadWidth; ix++) {
            float const px = std::abs(ix - p(0) - (PadWidth / 2)) / (W / 2.f);
            if (px > 1.f) {
              k(ix, iy) = 0.f;
            } else {
              float const kx = std::exp(beta * (std::sqrt(1.f - px * px) - 1.f));
              k(ix, iy) = kx * ky;
            }
          }
        }
      }
    } else if constexpr (N == 3) {
      for (size_t iz = 0; iz < PadWidth; iz++) {
        float const pz = std::abs((iz - p(2) - (PadWidth / 2))) / (W / 2.f);
        if (pz > 1.f) {
          for (size_t iy = 0; iy < PadWidth; iy++) {
            for (size_t ix = 0; ix < PadWidth; ix++) {
              k(ix, iy, iz) = 0.f;
            }
          }
        } else {
          float const kz = std::exp(beta * (std::sqrt(1.f - pz * pz) - 1.f));
          for (size_t iy = 0; iy < PadWidth; iy++) {
            float const py = std::abs(iy - p(1) - (PadWidth / 2)) / (W / 2.f);
            if (py > 1.f) {
              for (size_t ix = 0; ix < PadWidth; ix++) {
                k(ix, iy, iz) = 0.f;
              }
            } else {
              float const ky = std::exp(beta * (std::sqrt(1.f - py * py) - 1.f));
              for (size_t ix = 0; ix < PadWidth; ix++) {
                float const px = std::abs(ix - p(0) - (PadWidth / 2)) / (W / 2.f);
                if (px > 1.f) {
                  k(ix, iy, iz) = 0.f;
                } else {
                  float const kx = std::exp(beta * (std::sqrt(1.f - px * px) - 1.f));
                  k(ix, iy, iz) = scale * kx * ky * kz;
                }
              }
            }
          }
        }
      }
    }
    return k;
    // auto const z2 = this->distSq(p);
    // return (z2 > 1.f) // Strictly this should be (z2.sqrt() > 1.f) but numerical issues make this problematic
    //   .select(z2.constant(0.f), z2.constant(1.f) * (((z2.constant(1.f) - z2).sqrt() - 1.f) *
    //   z2.constant(beta)).exp());
  }
};

std::unique_ptr<NewKernel> make_new_kernel(std::string const &k, bool const is3D, float const os);

} // namespace rl
