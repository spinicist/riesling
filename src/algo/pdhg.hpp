#pragma once

#include "op/ops.hpp"
#include "prox/prox.hpp"

namespace rl {

struct PDHG
{
  using Op = Ops::Op<Cx>;
  using Scalar = typename Op::Scalar;
  using Vector = typename Op::Vector;
  using CMap = typename Op::CMap;
  
  std::shared_ptr<Op> A, P, G; // System, pre-conditioner, prox transform
  std::shared_ptr<Prox<Cx>> prox;
  Index iterLimit = 8;

  auto run(Cx const *bdata, float τ = 1.f /* Primal step size */) const -> Vector;
};

} // namespace rl
