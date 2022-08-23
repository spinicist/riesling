#pragma once

#include "common.hpp"
#include "log.h"
#include "precond/precond.hpp"
#include "tensorOps.h"
#include "threads.h"
#include "types.h"
#include "util.hpp"

namespace rl {
/* Based on https://github.com/PythonOptimizers/pykrylov/blob/master/pykrylov/lls/lsmr.py
 */

/*
 * LSMR with arbitrary regularization, i.e. Solve (A'A + λI)x = A'b + c with warm start
 */
template <typename Op>
struct LSMR
{
  using Input = typename Op::Input;
  using Output = typename Op::Output;

  Op &op;
  Precond<Output> *M = nullptr; // Left pre-conditioner
  Index iterLimit;
  float aTol = 1.e-6f;
  float bTol = 1.e-6f;
  float cTol = 1.e-6f;
  float const λ = 0.f;
  bool const debug = false;

  Input run(typename Op::Output const &b, Input const &x0 = Input(), Input const &xr = Input()) const
  {
    auto dev = Threads::GlobalDevice();
    // Allocate all memory

    auto const inDims = op.inputDimensions();
    auto const outDims = op.outputDimensions();

    // Workspace variables
    Output Mu(outDims), u(outDims);
    Input v(inDims), h(inDims), h̅(inDims), x(inDims), ur;

    CheckDimsEqual(b.dimensions(), outDims);
    if (x0.size()) {
      CheckDimsEqual(x0.dimensions(), inDims);
      x.device(dev) = x0;
      Mu.device(dev) = b - op.A(x);
    } else {
      x.setZero();
      Mu.device(dev) = b;
    }
    u.device(dev) = M ? M->apply(Mu) : Mu;
    float β;
    if (λ > 0.f) {
      ur.resize(inDims);
      if (xr.size()) {
        CheckDimsEqual(xr.dimensions(), inDims);
        ur.device(dev) = xr * xr.constant(sqrt(λ)) - x * x.constant(sqrt(λ));
      } else {
        ur.device(dev) = -x * x.constant(sqrt(λ));
      }
      β = sqrt(std::real(CheckedDot(u, Mu)) + Norm2(ur));
    } else {
      β = sqrt(std::real(CheckedDot(u, Mu)));
    }
    Mu.device(dev) = Mu / Mu.constant(β);
    u.device(dev) = u / u.constant(β);
    if (λ > 0.f) {
      ur.device(dev) = ur / ur.constant(β);
      v.device(dev) = op.Adj(u) + sqrt(λ) * ur;
    } else {
      v.device(dev) = op.Adj(u);
    }
    float α = Norm(v);
    v.device(dev) = v / v.constant(α);
    h.device(dev) = v;
    h̅.setZero();

    // Initialize transformation variables. There are a lot
    float ζ̅ = α * β;
    float α̅ = α;
    float ρ = 1;
    float ρ̅ = 1;
    float c̅ = 1;
    float s̅ = 0;

    // Initialize variables for ||r||
    float β̈ = β;
    float β̇ = 0;
    float ρ̇old = 1;
    float τ̃old = 0;
    float θ̃ = 0;
    float ζ = 0;

    // Initialize variables for estimation of ||A|| and cond(A)
    float normA2 = α * α;
    float maxρ̅ = 0;
    float minρ̅ = std::numeric_limits<float>::max();
    float const normb = β;

    if (debug) {
      Log::Tensor(v, "lsmr-v-init");
      Log::Tensor(x, "lsmr-x-init");
    }

    Log::Print(FMT_STRING("LSMR    α {:5.3E} β {:5.3E} λ {}{}"), α, β, λ, x0.size() ? " with initial guess" : "");

    for (Index ii = 0; ii < iterLimit; ii++) {
      // Bidiagonalization step
      Mu.device(dev) = op.A(v) - α * Mu;
      u.device(dev) = M ? M->apply(Mu) : Mu;
      if (λ > 0.f) {
        ur.device(dev) = (sqrt(λ) * v) - (α * ur);
        β = sqrt(std::real(CheckedDot(Mu, u)) + std::real(CheckedDot(ur, ur)));
      } else {
        β = sqrt(std::real(CheckedDot(Mu, u)));
      }
      Mu.device(dev) = Mu / Mu.constant(β);
      u.device(dev) = u / u.constant(β);
      if (λ > 0.f) {
        ur.device(dev) = ur / ur.constant(β);
        v.device(dev) = op.Adj(u) + (sqrt(λ) * ur) - (β * v);
      } else {
        v.device(dev) = op.Adj(u) - (β * v);
      }
      α = Norm(v);
      v.device(dev) = v / v.constant(α);

      // Construct rotation
      float ρold = ρ;
      float c, s;
      std::tie(c, s, ρ) = SymOrtho(α̅, β);
      float θnew = s * α;
      α̅ = c * α;

      // Use a plane rotation (Qbar_i) to turn R_i^T to R_i^bar
      float ρ̅old = ρ̅;
      float ζold = ζ;
      float θ̅ = s̅ * ρ;
      float ρtemp = c̅ * ρ;
      std::tie(c̅, s̅, ρ̅) = SymOrtho(c̅ * ρ, θnew);
      ζ = c̅ * ζ̅;
      ζ̅ = -s̅ * ζ̅;

      // Update h, h̅h, x.
      h̅.device(dev) = h - (θ̅ * ρ / (ρold * ρ̅old)) * h̅;
      x.device(dev) = x + (ζ / (ρ * ρ̅)) * h̅;
      h.device(dev) = v - (θnew / ρ) * h;

      if (debug) {
        Log::Tensor(x, fmt::format(FMT_STRING("lsmr-x-{:02d}"), ii));
        Log::Tensor(v, fmt::format(FMT_STRING("lsmr-v-{:02d}"), ii));
      }
      // Estimate of ||r||.
      // Apply rotation P{k-1}.
      float const β̂ = c * β̈;
      β̈ = -s * β̈;

      // Apply rotation Qtilde_{k-1}.
      // β̇ = β̇_{k-1} here.

      float const θ̃old = θ̃;
      auto [c̃old, s̃old, ρ̃old] = SymOrtho(ρ̇old, θ̅);
      θ̃ = s̃old * ρ̅;
      ρ̇old = c̃old * ρ̅;
      β̇ = -s̃old * β̇ + c̃old * β̂;

      // β̇   = β̇_k here.
      // ρ̇old = ρ̇_k  here.

      τ̃old = (ζold - θ̃old * τ̃old) / ρ̃old;
      float const τ̇ = (ζ - θ̃ * τ̃old) / ρ̇old;
      float const normr = std::sqrt(pow(β̇ - τ̇, 2.f) + β̈ * β̈);

      // Estimate ||A||.
      normA2 += β * β;
      float const normA = std::sqrt(normA2);
      normA2 += α * α;

      // Estimate cond(A).
      maxρ̅ = std::max(maxρ̅, ρ̅old);
      if (ii > 1) {
        minρ̅ = std::min(minρ̅, ρ̅old);
      }
      float const condA = std::max(maxρ̅, ρtemp) / std::min(minρ̅, ρtemp);

      // Convergence tests - go in pairs which check large/small values then the user tolerance
      float const normar = abs(ζ̅);
      float const normx = Norm(x);

      Log::Print(
        FMT_STRING("LSMR {:02d} α {:5.3E} β {:5.3E} |r| {:5.3E} |Ar| {:5.3E} |A| {:5.3E} cond(A) "
                   "{:5.3E} |x| {:5.3E}"),
        ii,
        α,
        β,
        normr,
        normar,
        normA,
        condA,
        normx);

      if (1.f + (1.f / condA) <= 1.f) {
        Log::Print(FMT_STRING("Cond(A) is very large"));
        break;
      }
      if ((1.f / condA) <= cTol) {
        Log::Print(FMT_STRING("Cond(A) has exceeded limit"));
        break;
      }

      if (1.f + (normar / (normA * normr)) <= 1.f) {
        Log::Print(FMT_STRING("Least-squares solution reached machine precision"));
        break;
      }
      if ((normar / (normA * normr)) <= aTol) {
        Log::Print(FMT_STRING("Least-squares = {:5.3E} < aTol = {:5.3E}"), normar / (normA * normr), aTol);
        break;
      }

      if (normr <= (bTol * normb + aTol * normA * normx)) {
        Log::Print(FMT_STRING("Ax - b <= aTol, bTol"));
        break;
      }
      if ((1.f + normr / (normb + normA * normx)) <= 1.f) {
        Log::Print(FMT_STRING("Ax - b reached machine precision"));
        break;
      }
    }

    return x;
  }
};

} // namespace rl
