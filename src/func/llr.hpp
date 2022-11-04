#pragma once

#include "functor.hpp"

namespace rl {

struct LLR final : Prox<Cx4>
{
  float λ;
  Index patchSize, windowSize;
  LLR(float, Index, Index);

  auto operator()(float const α, Eigen::TensorMap<Cx4 const> x) const -> Cx4;
};

} // namespace rl
