#pragma once

#include "prox.hpp"

namespace rl::Proxs {

struct SLR final : Prox<Cx>
{
  PROX_INHERIT(Cx)
  SLR(float const, Index const, Sz5 const);
  void apply(float const α, CMap const &x, Map &z) const;

private:
  float                           λ;
  Index                           kSz;
  Sz5                             shape;
};

} // namespace rl::Proxs
