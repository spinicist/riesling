#pragma once

#include "op/ops.hpp"
#include <functional>
#include <span>

namespace rl {
/* Based on https://github.com/PythonOptimizers/pykrylov/blob/master/pykrylov/lls/lsmr.py
 */

/*
 * LSMR with arbitrary regularization, i.e. Solve (A'A + λI)x = A'b + c with warm start
 */
struct LSMR
{
  using Op = Ops::Op<Cx>;
  using Vector = typename Op::Vector;
  using Map = typename Op::Map;
  using CMap = typename Op::CMap;

  Op::Ptr op;
  Op::Ptr M = nullptr; // Pre-conditioner
  Index   iterLimit = 4;
  float   aTol = 1.e-6f;
  float   bTol = 1.e-6f;
  float   cTol = 1.e-6f;

  std::function<void(Index const iter, Vector const &)> debug = nullptr;

  auto run(Vector const &b, float const λ = 0.f, Vector const &x0 = Vector()) const -> Vector;
  auto run(CMap const b, float const λ = 0.f, CMap x0 = CMap(nullptr, 0)) const -> Vector;
};

} // namespace rl
