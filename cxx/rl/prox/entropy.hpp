#pragma once

#include "prox.hpp"

namespace rl::Proxs {

struct Entropy final : Prox<Cx>
{
  PROX_INHERIT(Cx)
  Entropy(float const λ, Index const sz);
  void apply(float const α, CMap const x, Map z) const;

private:
  float λ;
};

struct NMREntropy final : Prox<Cx>
{
  PROX_INHERIT(Cx)
  NMREntropy(float const λ, Index const sz);
  void apply(float const α, CMap const x, Map z) const;

private:
  float λ;
};

} // namespace rl::Proxs
