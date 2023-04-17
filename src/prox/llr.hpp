#pragma once

#include "prox.hpp"

namespace rl {

struct LLR final : Prox<Cx>
{
  float λ;
  Index patchSize, windowSize;
  Sz4 shape;
  LLR(float const, Index const, Index const, Sz4 const);

  void operator()(float const α, Vector const &x, Vector &z) const;
};

} // namespace rl
