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
    Index iters0 = 16;   // Number of inner iterations on first outer iteration
    Index iters1 = 8;    // Number of inner iterations on subsequent outer iterations
    float aTol = 1.e-6f; // LSMR tolerance parameters
    float bTol = 1.e-6f;
    float cTol = 1.e-6f;

    float ε = 1.e-3f;      // Convergence criteria
    Index outerLimit = 64; // Number of outer iterations
    Index restart = 8;     // Restart average every N iterations

    float ρ = 1;    // Initial penalty parameter
    bool  updateρ;  // Update ρ when average restarts
    float τ = 10.f; // Fallback ρ update
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
  Op::Ptr                  Aʹ, Minvʹ;

  Vector mutable x, x_k, bʹ;
  std::vector<float> mutable ρ;
  std::vector<Ops::DiagScale::Ptr> mutable ρops;
  std::vector<Vector> mutable z, y;

  // Helper function
  auto x_update(Index const) const -> bool;
  void zy_update(Index const, Index const, Vector const &) const;
};

} // namespace rl
