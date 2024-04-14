#pragma once

#include "prox.hpp"

namespace rl::Proxs {

struct L1 final : Prox<Cx>
{
  PROX_INHERIT(Cx)
  float λ;

  L1(float const λ, Index const sz);
  void apply(float const α, CMap const &x, Map &z) const;
  void apply(std::shared_ptr<Op> const α, CMap const &x, Map &z) const;
};

struct L2 final : Prox<Cx>
{
  PROX_INHERIT(Cx)
  float λ;
  Index block;

  L2(float const λ, Index const sz, Index const blk);
  void apply(float const α, CMap const &x, Map &z) const;
  void apply(std::shared_ptr<Op> const α, CMap const &x, Map &z) const;
};

} // namespace rl::Proxs
