#pragma once

#include "op/ops.hpp"
#include "prox.hpp"

namespace rl::Proxs {

template <typename Scalar>
struct L2 final : Prox<Scalar>
{
  PROX_INHERIT(Scalar)
  float λ;
  CMap  y;

  L2(float const λ, Index const sz);
  L2(float const λ, CMap const bias);
  void apply(float const α, CMap const &x, Map &z) const;
  void apply(std::shared_ptr<Ops::Op<Scalar>> const α, CMap const &x, Map &z) const;
  void setBias(Scalar const *data);
};

} // namespace rl::Proxs