#pragma once

#include "common.hpp"
#include "log.h"
#include "precond/precond.hpp"
#include "tensorOps.hpp"
#include "threads.hpp"
#include "types.h"
#include "util.hpp"

namespace rl {
/* Based on https://github.com/PythonOptimizers/pykrylov/blob/master/pykrylov/lls/lsqr.py
 */

/*
 * LSQR with arbitrary regularization, i.e. Solve (A'A + λI)x = A'b + c with warm start
 */
template <typename Op>
struct LSQR
{
  using Input = typename Op::Input;
  using Output = typename Op::Output;

  Op &op;
  Precond<Output> *M = nullptr; // Left pre-conditioner
  Precond<Input> *N = nullptr;  // Right pre-conditioner
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
    Input x(inDims), Nv(inDims), v(inDims), w(inDims), ur;
    Mu.setZero();
    u.setZero();
    x.setZero();
    Nv.setZero();
    v.setZero();
    w.setZero();

    CheckDimsEqual(b.dimensions(), outDims);
    if (x0.size()) {
      CheckDimsEqual(x0.dimensions(), inDims);
      x.device(dev) = x0;
      Mu.device(dev) = b - op.forward(x);
    } else {
      x.setZero();
      Mu.device(dev) = b;
    }
    u.device(dev) = M ? M->apply(Mu) : Mu;
    float β;
    if (λ > 0) {
      ur.resize(inDims);
      if (xr.size()) {
        CheckDimsEqual(xr.dimensions(), inDims);
        ur.device(dev) = xr * xr.constant(sqrt(λ)) - x * x.constant(sqrt(λ));
      } else {
        ur.device(dev) = -x * x.constant(sqrt(λ));
      }
      β = std::sqrt(CheckedDot(u, Mu) + CheckedDot(ur, ur));
    } else {
      β = std::sqrt(CheckedDot(u, Mu));
    }
    float const normb = β; // For convergence tests
    Mu.device(dev) = Mu / Mu.constant(β);
    u.device(dev) = u / u.constant(β);
    if (λ > 0.f) {
      ur.device(dev) = ur / ur.constant(β);
      Nv.device(dev) = op.adjoint(u) + sqrt(λ) * ur;
    } else {
      Nv.device(dev) = op.adjoint(u);
    }
    v.device(dev) = N ? N->apply(Nv) : Nv;
    float α = std::sqrt(CheckedDot(v, Nv));
    Nv.device(dev) = Nv / Nv.constant(α);
    v.device(dev) = v / v.constant(α);
    w.device(dev) = v; // Test if this should be Nv

    float ρ̅ = α;
    float ɸ̅ = β;
    float ddnorm = 0;
    float normA = 0;

    if (debug) {
      Log::Tensor(x, "lsqr-x-init");
      Log::Tensor(v, "lsqr-v-init");
    }

    Log::Print(FMT_STRING("LSQR    α {:5.3E} β {:5.3E} λ {}{}"), α, β, λ, x0.size() ? " with initial guess" : "");

    for (Index ii = 0; ii < iterLimit; ii++) {
      // Bidiagonalization step
      Mu.device(dev) = op.forward(v) - α * Mu;
      u.device(dev) = M ? M->apply(Mu) : Mu;
      if (λ > 0.f) {
        ur.device(dev) = (sqrt(λ) * Nv) - (α * ur);
        β = std::sqrt(CheckedDot(Mu, u) + CheckedDot(ur, ur));
      } else {
        β = std::sqrt(CheckedDot(Mu, u));
      }
      Mu.device(dev) = Mu / Mu.constant(β);
      u.device(dev) = u / u.constant(β);
      if (λ > 0.f) {
        ur.device(dev) = ur / ur.constant(β);
        Nv.device(dev) = op.adjoint(u) + (sqrt(λ) * ur) - (β * Nv);
      } else {
        Nv.device(dev) = op.adjoint(u) - (β * Nv);
      }
      v.device(dev) = N ? N->apply(Nv) : Nv;
      α = std::sqrt(CheckedDot(v, Nv));

      Nv.device(dev) = Nv / Nv.constant(α);
      v.device(dev) = v / v.constant(α);

      float const ρ = std::sqrt(ρ̅ * ρ̅ + β * β);
      float const cs = ρ̅ / ρ;
      float const sn = β / ρ;
      float const θ = sn * α;
      ρ̅ = -cs * α;
      float const ɸ = cs * ɸ̅;
      ɸ̅ = sn * ɸ̅;
      ddnorm = ddnorm + Norm2(w) / (ρ * ρ);
      x.device(dev) = x + w * w.constant(ɸ / ρ);
      w.device(dev) = v - w * w.constant(θ / ρ);

      if (debug) {
        Log::Tensor(x, fmt::format(FMT_STRING("lsqr-x-{:02d}"), ii));
        Log::Tensor(v, fmt::format(FMT_STRING("lsqr-v-{:02d}"), ii));
      }

      // Estimate norms
      float const normx = Norm(x);
      float const normr = ɸ̅;
      float const normAr = ɸ̅ * α * std::abs(cs);
      normA = std::sqrt(normA * normA + α * α + β * β);
      float const condA = normA * std::sqrt(ddnorm);

      Log::Print(
        FMT_STRING("LSQR {:02d} α {:5.3E} β {:5.3E} |r| {:5.3E} |Ar| {:5.3E} |A| {:5.3E} cond(A) "
                   "{:5.3E} |x| {:5.3E}"),
        ii,
        α,
        β,
        normr,
        normAr,
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

      if (1.f + (normAr / (normA * normr)) <= 1.f) {
        Log::Print(FMT_STRING("Least-squares solution reached machine precision"));
        break;
      }
      if ((normAr / (normA * normr)) <= aTol) {
        Log::Print(FMT_STRING("Least-squares = {:5.3E} < {:5.3E}"), normAr / (normA * normr), aTol);
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

    return N ? N->inv(x) : x;
  }
};

} // namespace rl
