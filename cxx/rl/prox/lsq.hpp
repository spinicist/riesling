#pragma once

#include "../op/ops.hpp"
#include "prox.hpp"

namespace rl::Proxs {

template <typename Scalar = Cx>
struct LeastSquares final : Prox<Scalar>
{
  PROX_INHERIT(Scalar)
  using Ptr = std::shared_ptr<LeastSquares>;

  float λ;
  CMap  y;

  LeastSquares(float const λ, Index const sz);
  LeastSquares(float const λ, CMap bias);
  void apply(float const α, CMap x, Map z) const;
  void apply(std::shared_ptr<Ops::Op<Scalar>> const α, CMap x, Map z) const;
  void setY(CMap y);
};

} // namespace rl::Proxs