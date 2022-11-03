#pragma once

#include "functor.hpp"

namespace rl {

struct LLR final : Prox<Cx4>
{
  Index patchSize;
  bool notSliding;
  LLR(Index, bool);

  auto operator()(float const λ, Eigen::TensorMap<Cx4 const> x) const -> Cx4;

private:
  auto applySliding(float const λ, Eigen::TensorMap<Cx4 const> x) const -> Cx4;
  auto applyFixed(float const λ, Eigen::TensorMap<Cx4 const> x) const -> Cx4;
  Cx4 y;
};

} // namespace rl
