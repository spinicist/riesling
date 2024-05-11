#pragma once

#include "func/functor.hpp"
#include "log.hpp"
#include "op/ops.hpp"
#include "signals.hpp"
#include "tensors.hpp"
#include "threads.hpp"

namespace rl {

struct LAD
{
  using Op = Ops::Op<Cx>;
  using Vector = typename Op::Vector;
  using Map = typename Op::Map;
  using CMap = typename Op::CMap;

  std::shared_ptr<Op> A; // Forward model
  std::shared_ptr<Op> M; // Pre-conditioner

  Index iters0 = 4;
  Index iters1 = 1;
  float aTol = 1.e-6f;
  float bTol = 1.e-6f;
  float cTol = 1.e-6f;

  Index outerLimit;
  float ε;
  float μ;
  float τmax;

  auto run(Cx const *bdata, float ρ) const -> Vector;
};

} // namespace rl
