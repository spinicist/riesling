#pragma once

#include "../op/ops.hpp"
#include "../prox/prox.hpp"
#include "../prox/stack.hpp"
#include "regularizer.hpp"

namespace rl::PDHG {
using Op = Ops::Op;
using Prox = Proxs::Prox;
using Vector = typename Op::Vector;
using CMap = typename Op::CMap;
using Debug = std::function<void(Index const, Vector const &, Vector const &)>;
struct Opts
{
  bool  lad;
  Index imax;
  float deltaTol;
  float λA;
};

auto Run(Vector const &b, Op::Ptr A, Op::Ptr P, std::vector<Regularizer> const &regs, Opts opts, Debug d = nullptr) -> Vector;
auto Run(CMap b, Op::Ptr A, Op::Ptr P, std::vector<Regularizer> const &regs, Opts opts, Debug d = nullptr) -> Vector;
} // namespace rl::PDHG
