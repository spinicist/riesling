#pragma once

#include "func/functor.hpp"
#include "op/operator.hpp"
#include "log.hpp"
#include "signals.hpp"
#include "tensorOps.hpp"
#include "threads.hpp"

namespace rl {

struct ADMM
{
  using Op = Operator<Cx>;
  using Vector = typename Op::Vector;
  using Map = typename Op::Map;

  std::shared_ptr<Op> lsq; // Operator for least-squares
  std::shared_ptr<Op> M; // Left pre-conditioner
  Index lsqLimit = 8;
  float aTol = 1.e-6f;
  float bTol = 1.e-6f;
  float cTol = 1.e-6f;

  std::vector<std::shared_ptr<Operator<Cx>>> reg_ops;
  std::vector<std::shared_ptr<Prox<Cx>>> prox;
  Index outterLimit = 8;
  float α = 1.f;  // Over-relaxation
  float μ = 10.f; // Primal-dual mismatch limit
  float τ = 2.f;  // Primal-dual mismatch rescale
  float abstol = 1.e-4f;
  float reltol = 1.e-4f;

  auto run(Cx *bdata, float ρ) const -> Vector;
};

} // namespace rl
