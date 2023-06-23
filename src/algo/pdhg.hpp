#pragma once

#include "op/ops.hpp"
#include "prox/prox.hpp"

namespace rl {

struct PDHG
{
  using Op = Ops::Op<Cx>;
  using Prox = Proxs::Prox<Cx>;
  using Scalar = typename Op::Scalar;
  using Vector = typename Op::Vector;
  using CMap = typename Op::CMap;

  std::shared_ptr<Op> A, P; // System, preconditioner, prox transform
  std::vector<std::shared_ptr<Op>> G; // Prox transforms
  std::vector<std::shared_ptr<Prox>> prox;
  std::vector<float> σG; // Step-size/precond for prox transforms

  Index iterLimit = 8;

  std::function<void(Index const, Vector const &, Vector const &, Vector const &)> debug = nullptr;

  auto run(Cx const *bdata, float τ) const -> Vector;
};

} // namespace rl
