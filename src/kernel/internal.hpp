#pragma once

#include "types.h"

namespace rl {

template <size_t ND, size_t W>
struct KernelTypes
{
  using OneD = Eigen::TensorFixedSize<float, Eigen::Sizes<W>>;
  // Welcome to C++. Declares a TensorFixedSize<float, Eigen::Sizes<W, W, ...>>
  using Tensor = typename decltype([]<std::size_t... Is>(std::index_sequence<Is...>) {
    return std::type_identity<Eigen::TensorFixedSize<float, Eigen::Sizes<(Is, W)...>>>();
  }(std::make_index_sequence<ND>()))::type;
  using Point = Eigen::Matrix<float, ND, 1>;
};

template <size_t W>
inline constexpr auto Centers()
{
  Eigen::TensorFixedSize<float, Eigen::Sizes<W>> pos;
  for (size_t ii = 0; ii < W; ii++) {
    pos(ii) = ii + 0.5f - (W / 2.f);
  }
  return pos;
}

} // namespace rl
