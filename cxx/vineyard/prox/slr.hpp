#pragma once

#include "prox.hpp"

namespace rl::Proxs {

template <int ND> struct SLR final : Prox<Cx>
{
  PROX_INHERIT(Cx)
  SLR(float const, Sz<ND> const);
  void apply(float const α, CMap const &x, Map &z) const;

private:
  float  λ;
  Sz<ND> shape;
};

} // namespace rl::Proxs
