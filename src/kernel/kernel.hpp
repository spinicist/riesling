#pragma once

#include "types.hpp"

namespace rl {

template <size_t ND, size_t W>
struct KernelSizes
{
};

template <size_t W>
struct KernelSizes<1, W>
{
  using Type = Eigen::Sizes<W>;
};

template <size_t W>
struct KernelSizes<2, W>
{
  using Type = Eigen::Sizes<W, W>;
};

template <size_t W>
struct KernelSizes<3, W>
{
  using Type = Eigen::Sizes<W, W, W>;
};

template <size_t ND, size_t W>
struct Kernel
{
  using OneD = Eigen::TensorFixedSize<float, Eigen::Sizes<W>>;
  using Tensor = Eigen::TensorFixedSize<float, typename KernelSizes<ND, W>::Type>;
  using Point = Eigen::Matrix<float, ND, 1>;

  OneD centers;

  Kernel()
  {
    Eigen::TensorFixedSize<float, Eigen::Sizes<W>> pos;
    for (size_t ii = 0; ii < W; ii++) {
      centers(ii) = ii + 0.5f - (W / 2.f);
    }
  }

  virtual auto operator()(Point const p) const -> Tensor = 0;
};

} // namespace rl
