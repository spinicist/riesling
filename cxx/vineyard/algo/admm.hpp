#pragma once

#include "op/ops.hpp"
#include "prox/prox.hpp"

namespace rl {

struct ADMM
{
  using Op = Ops::Op<Cx>;
  using Vector = typename Op::Vector;
  using Map = typename Op::Map;
  using CMap = typename Op::CMap;
  using DebugX = std::function<void(Index const, Vector const &)>;
  using DebugZ = std::function<void(Index const, Index const, Vector const &, Vector const &, Vector const &)>;

  std::shared_ptr<Op> A; // Forward model
  std::shared_ptr<Op> M; // Pre-conditioner

  std::vector<std::shared_ptr<Op>>              reg_ops;
  std::vector<std::shared_ptr<Proxs::Prox<Cx>>> prox;

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

  auto run(Cx const *bdata, float const ρ) const -> Vector;
};

} // namespace rl
