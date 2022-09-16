#pragma once

#include "types.hpp"

namespace rl {

template <size_t ND, size_t W>
struct Kernel
{
  using OneD = Eigen::TensorFixedSize<float, Eigen::Sizes<W>>;
  // Welcome to C++. Declares a TensorFixedSize<float, Eigen::Sizes<W, W, ...>>
  using Tensor = typename decltype([]<std::size_t... Is>(std::index_sequence<Is...>) {
    return std::type_identity<Eigen::TensorFixedSize<float, Eigen::Sizes<(Is, W)...>>>();
  }(std::make_index_sequence<ND>()))::type;
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
