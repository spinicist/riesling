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

  std::shared_ptr<Op> A; // Op for least-squares
  std::shared_ptr<Op> M; // Pre-conditioner
  Index lsqLimit = 8;
  float aTol = 1.e-6f;
  float bTol = 1.e-6f;
  float cTol = 1.e-6f;

  std::vector<std::shared_ptr<Op>> reg_ops;
  std::vector<std::shared_ptr<Prox<Cx>>> prox;
  Index outerLimit = 8;
  float α = 1.f;  // Over-relaxation
  float μ = 10.f; // Primal-dual mismatch limit
  float τ = 2.f;  // Primal-dual mismatch rescale
  float abstol = 1.e-4f;
  float reltol = 1.e-4f;
  bool hogwild = false;

  std::function<void(Index const, Vector const &)> debug_x = nullptr;
  std::function<void(Index const, Index const, ADMM::Vector const &)> debug_z = nullptr;

  auto run(Cx const *bdata, float ρ) const -> Vector;
};

} // namespace rl
