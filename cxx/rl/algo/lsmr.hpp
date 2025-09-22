#pragma once

#include "../op/ops.hpp"
#include <functional>

namespace rl {
/* Based on https://github.com/PythonOptimizers/pykrylov/blob/master/pykrylov/lls/lsmr.py
 */

/*
 * LSMR with arbitrary regularization, i.e. Solve (A'A + λI)x = A'b + c with warm start
 */
struct LSMR
{
  using Op = Ops::Op;
  using Vector = typename Op::Vector;
  using Map = typename Op::Map;
  using CMap = typename Op::CMap;
  using DbgFunc = typename std::function<void(Index const iter, Vector const &, Vector const &)>;

  struct Opts
  {
    Index imax = 4;
    float aTol = 1.e-6f;
    float bTol = 1.e-6f;
    float cTol = 1.e-6f;
    float λ = 0.f;
  };

  Op::Ptr A;
  Op::Ptr Minv = nullptr; // Left Pre-conditioner
  Op::Ptr Ninv = nullptr; // Right Pre-conditioner

  Opts    opts;
  DbgFunc debug = nullptr;

  auto run(Vector const &b, Vector const &x0 = Vector()) const -> Vector;
  auto run(CMap b, CMap x0 = CMap(nullptr, 0)) const -> Vector;
};

} // namespace rl
