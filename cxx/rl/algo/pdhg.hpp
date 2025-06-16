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
  using Debug = std::function<void(Index const, Vector const &, Vector const &, Vector const &)>;

  PDHG(Op::Ptr                         A,
       Op::Ptr                         P,
       std::vector<Regularizer> const &regs,
       float                           λA = 0.f,
       float                           λG = 0.f,
       Index                           imax = 4,
       float                           resTol = 1.e-6f,
       float                           deltaTol = 1.e-6f,
       Debug                           d = nullptr);

  auto run(Vector const &b) const -> Vector;
  auto run(CMap b) const -> Vector;

  Op::Ptr              A, P, G;
  Proxs::Prox<Cx>::Ptr proxʹ;

  Index imax;
  float resTol, deltaTol;
  float σ, τ, θ;

  Debug debug;
};

} // namespace rl
