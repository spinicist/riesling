#pragma once

#include "types.hpp"

namespace rl {

template <typename T>
struct Prox
{
  using Input = Eigen::TensorMap<T const>;
  using Output = T;

  virtual auto operator()(float const λ, Input in) const -> T = 0;
  virtual ~Prox(){};
};

template <typename T>
struct IdentityProx final : Prox<T>
{
  auto operator()(float const λ, Eigen::TensorMap<T const> in) const -> T { return in; }
};

}
