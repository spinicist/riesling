#pragma once

#include "../op/ops.hpp"
#include "../prox/prox.hpp"
#include "../prox/stack.hpp"
#include "regularizer.hpp"

namespace rl {

/* Follow Sidky 2011 */
struct PDHG
{
  using Op = Ops::Op;
  using Prox = Proxs::Prox;
  using Vector = typename Op::Vector;
  using CMap = typename Op::CMap;
  using Debug = std::function<void(Index const, Vector const &, Vector const &, Vector const &)>;

  struct Opts
  {
    Index imax;
    float resTol, deltaTol;
    float λA, λG;
  };

  PDHG(Op::Ptr A, Op::Ptr P, Proxs::Prox::Ptr proxF, std::vector<Regularizer> const &regs, Opts opts, Debug d = nullptr);

  auto run() const -> Vector;

  Op::Ptr          K, P;
  Proxs::Prox::Ptr proxF;
  Index            imax;
  float            resTol, deltaTol;
  float            σ, τ, θ;
  Debug            debug;
};

} // namespace rl
