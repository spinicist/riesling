#pragma once

#include "../op/ops.hpp"
#include "regularizer.hpp"

namespace rl {

struct ADMM
{
  using Op = Ops::Op;
  using Vector = typename Op::Vector;
  using Map = typename Op::Map;
  using CMap = typename Op::CMap;
  using DebugX = std::function<void(Index const, Vector const &)>;
  using DebugZ = std::function<void(Index const, Index const, Vector const &, Vector const &, Vector const &)>;

  struct Opts
  {
    Index iters0 = 4;    // Number of inner iterations on first outer iteration
    Index iters1 = 1;    // Number of inner iterations on subsequent outer iterations
    float aTol = 1.e-6f; // LSMR tolerance parameters
    float bTol = 1.e-6f;
    float cTol = 1.e-6f;

    Index outerLimit = 64; // Number of outer iterations
    float ε = 1.e-3f;      // Combined apply/dual tolerance parameter
    float ρ = 1;           // Penalty parameter
    bool  balance = true;  // Apply residual balancing scheme
    float μ = 1.2f;        // Residual balancing tolerance
    float τmax = 10.f;     // Maximum residual balancing ratio
    float ɑ = 0.f;         // Over-relaxation parameter, set 1 < ɑ < 2
  };

  ADMM(Op::Ptr A, Op::Ptr Minv, std::vector<Regularizer> const &regs, Opts opts, DebugX dx = nullptr, DebugZ dz = nullptr);
  auto run(Vector const &b) const -> Vector;
  auto run(CMap b) const -> Vector;

private:
  Op::Ptr                  A;    // Forward model
  Op::Ptr                  Minv; // Pre-conditioner
  std::vector<Regularizer> regs;
  Opts                     opts;
  DebugX                   debug_x = nullptr;
  DebugZ                   debug_z = nullptr;
  std::vector<Vector> mutable z, u;
  std::vector<Ops::DiagScale::Ptr> mutable ρscalers;
  Op::Ptr Aʹ, Minvʹ;
};

} // namespace rl
