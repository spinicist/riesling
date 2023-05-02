#pragma once

#include "bidiag.hpp"
#include "common.hpp"
#include "func/functor.hpp"
#include "log.hpp"
#include "op/operator.hpp"
#include "signals.hpp"
#include "tensorOps.hpp"
#include "threads.hpp"
#include "types.hpp"

namespace rl {
/* Based on https://github.com/PythonOptimizers/pykrylov/blob/master/pykrylov/lls/lsmr.py
 */

/*
 * LSMR with arbitrary regularization, i.e. Solve (A'A + λI)x = A'b + c with warm start
 */
struct LSMR
{
  using Op = LinOps::Op<Cx>;
  using Vector = typename Op::Vector;
  using Map = typename Op::Map;

  std::shared_ptr<Op> op;
  std::shared_ptr<Op> M, N; // Pre-conditioners
  Index iterLimit = 8;
  float aTol = 1.e-6f;
  float bTol = 1.e-6f;
  float cTol = 1.e-6f;
  bool debug = false;

  auto run(Cx *bdata, float const λ = 0.f, Cx *x0 = nullptr) const -> Vector;
};

} // namespace rl
