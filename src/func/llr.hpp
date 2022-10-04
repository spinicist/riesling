#pragma once

#include "functor.hpp"

namespace rl {

struct LLR final : Prox<Cx4>
{
  Index patchSize;
  bool sliding;
  LLR(Index, bool);

  auto operator()(float const λ, Cx4 const &) const -> Cx4;

private:
  auto applySliding(float const λ, Cx4 const &) const -> Cx4;
  auto applyFixed(float const λ, Cx4 const &) const -> Cx4;
};

} // namespace rl
