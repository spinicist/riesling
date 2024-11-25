#pragma once

#include "../op/ops.hpp"
#include "regularizer.hpp"

namespace rl {

struct ADMM
{
  using Op = Ops::Op<Cx>;
  using Vector = typename Op::Vector;
  using Map = typename Op::Map;
  using CMap = typename Op::CMap;
  using DebugX = std::function<void(Index const, Vector const &)>;
  using DebugZ = std::function<void(Index const, Index const, Vector const &, Vector const &, Vector const &)>;

  Op::Ptr A; // Forward model
  Op::Ptr Minv; // Pre-conditioner

  std::vector<Regularizer> regs;

  Index iters0 = 4;
  Index iters1 = 1;
  float aTol = 1.e-6f;
  float bTol = 1.e-6f;
  float cTol = 1.e-6f;

  Index outerLimit;
  float ε;
  float μ;
  float τmax;

  DebugX debug_x = nullptr;
  DebugZ debug_z = nullptr;

  auto run(Vector const &b, float const ρ) const -> Vector;
  auto run(CMap const b, float const ρ) const -> Vector;
};

} // namespace rl
