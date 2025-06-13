#pragma once

#include "../op/ops.hpp"
#include "../prox/prox.hpp"
#include "../prox/stack.hpp"
#include "regularizer.hpp"

namespace rl {

struct PDHG
{
  using Op = Ops::Op<Cx>;
  using Prox = Proxs::Prox<Cx>;
  using Scalar = typename Op::Scalar;
  using Vector = typename Op::Vector;
  using CMap = typename Op::CMap;
  using Callback = std::function<void(Index const, Vector const &, Vector const &, Vector const &)>;

  PDHG(std::shared_ptr<Op>             A,
       std::shared_ptr<Op>             P,
       std::vector<Regularizer> const &regs,
       Index const                     imax = 4,
       float const                     σ = 0.f,
       float const                     τ = 0.f,
       float const                     θ = 0.f);

  auto run(Vector const &b) const -> Vector;
  auto run(CMap b) const -> Vector;

private:
  Op::Ptr              A, P, G;
  Proxs::Prox<Cx>::Ptr proxʹ;

  Index imax;
  float σ, τ, θ;
};

} // namespace rl
