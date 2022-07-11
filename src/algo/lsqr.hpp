#pragma once

#include "common.hpp"
#include "log.h"
#include "tensorOps.h"
#include "threads.h"
#include "types.h"

/* Based on https://github.com/PythonOptimizers/pykrylov/blob/master/pykrylov/lls/lsqr.py
 */

/*
 * LSQR with arbitrary regularization, i.e. Solve (A'A + λI)x = A'b + c with warm start
 */
template <typename Op, typename LeftPrecond>
typename Op::Input lsqr(
  Index const &max_its,
  Op &op,
  typename Op::Output const &b,
  float const atol = 1.e-6f,
  float const btol = 1.e-6f,
  float const ctol = 1.e-6f,
  float const λ = 0.f,
  LeftPrecond const *M = nullptr, // Left preconditioner
  bool const debug = false,
  typename Op::Input const &x0 = typename Op::Input(),
  typename Op::Input const &xr = typename Op::Input())
{
  auto dev = Threads::GlobalDevice();
  // Allocate all memory
  using TI = typename Op::Input;
  using TO = typename Op::Output;
  auto const inDims = op.inputDimensions();
  auto const outDims = op.outputDimensions();

  // Workspace variables
  TO Mu(outDims), u(outDims);
  TI x(inDims), v(inDims), w(inDims), ur;

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
  if (λ > 0) {
    ur.resize(inDims);
    if (xr.size()) {
      CheckDimsEqual(xr.dimensions(), inDims);
      ur.device(dev) = xr * xr.constant(sqrt(λ)) - x * x.constant(sqrt(λ));
    } else {
      ur.device(dev) = -x * x.constant(sqrt(λ));
    }
    β = sqrt(std::real(Dot(u, Mu)) + Norm2(ur));
  } else {
    β = sqrt(std::real(Dot(u, Mu)));
  }
  float const normb = β; // For convergence tests
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
  w.device(dev) = v;

  float ρ̅ = α;
  float ɸ̅ = β;
  float ddnorm = 0;
  float normA = 0;

  if (debug) {
    Log::Tensor(v, "lsqr-v-init");
    Log::Tensor(x, "lsqr-x-init");
    if (ur.size()) {
      Log::Tensor(ur, "lsqr-ur-init");
    }
  }

  Log::Print(FMT_STRING("LSQR    α {:5.3E} β {:5.3E} λ {}{}"), α, β, λ, x0.size() ? " with initial guess" : "");

  for (Index ii = 0; ii < max_its; ii++) {
    // Bidiagonalization step
    Mu.device(dev) = op.A(v) - α * Mu;
    u.device(dev) = M ? M->apply(Mu) : Mu;
    if (λ > 0.f) {
      ur.device(dev) = (sqrt(λ) * v) - (α * ur);
      β = sqrt(std::real(Dot(Mu, u)) + std::real(Dot(ur, ur)));
    } else {
      β = sqrt(std::real(Dot(Mu, u)));
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
      Log::Tensor(w, fmt::format(FMT_STRING("lsqr-w-{:02d}"), ii));
      if (ur.size()) {
        Log::Tensor(ur, fmt::format(FMT_STRING("lsqr-ur-{:02d}"), ii));
      }
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
    if ((1.f / condA) <= ctol) {
      Log::Print(FMT_STRING("Cond(A) has exceeded limit"));
      break;
    }

    if (1.f + (normAr / (normA * normr)) <= 1.f) {
      Log::Print(FMT_STRING("Least-squares solution reached machine precision"));
      break;
    }
    if ((normAr / (normA * normr)) <= atol) {
      Log::Print(FMT_STRING("Least-squares = {:5.3E} < {:5.3E}"), normAr / (normA * normr), atol);
      break;
    }

    if (normr <= (btol * normb + atol * normA * normx)) {
      Log::Print(FMT_STRING("Ax - b <= atol, btol"));
      break;
    }
    if ((1.f + normr / (normb + normA * normx)) <= 1.f) {
      Log::Print(FMT_STRING("Ax - b reached machine precision"));
      break;
    }
  }

  return x;
}
