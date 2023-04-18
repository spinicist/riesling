#pragma once

#include "prox.hpp"

namespace rl {

struct LLR final : Prox<Cx>
{
  PROX_INHERIT(Cx)
  float λ;
  Index patchSize, windowSize;
  Sz4 shape;
  LLR(float const, Index const, Index const, Sz4 const);

  void apply(float const α, CMap const &x, Map &z) const;
};

} // namespace rl
