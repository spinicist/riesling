#pragma once

#include "log.h"
#include "tensorOps.h"
#include "threads.h"
#include "types.h"

/*
 * Wrapper for solving normal equations
 */
template <typename Op>
struct NormalEqOp
{
  using Input = typename Op::Input;
  Op const &op;

  auto inputDimensions() const
  {
    return op.inputDimensions();
  }

  Input A(typename Op::Input const &x) const
  {
    return Input(op.AdjA(x));
  }
};

template <typename Dims>
void CheckDims(Dims const a, Dims const b)
{
  if (a != b) {
    Log::Fail(FMT_STRING("Dimensions mismatch {} != {}"), a, b);
  }
}

/*
 * Conjugate gradients
 */
template <typename Op>
typename Op::Input cg(
  Index const &max_its,
  float const &thresh,
  Op const &op,
  typename Op::Input const &b,
  typename Op::Input const &x0 = typename Op::Input())
{
  float const scale = Norm(b);
  Log::Print(FMT_STRING("Starting Conjugate Gradients, scale {} threshold {}"), scale, thresh);
  auto dev = Threads::GlobalDevice();
  // Allocate all memory
  using T = typename Op::Input;
  auto const dims = op.inputDimensions();
  CheckDims(dims, b.dimensions());
  T q(dims), p(dims), r(dims), x(dims);
  // If we have an initial guess, use it
  if (x0.size()) {
    CheckDims(dims, x0.dimensions());
    Log::Print("Initialising CG with initial guess");
    r.device(dev) = (b - op.A(x0)) / r.constant(scale);
    x.device(dev) = x0 / x.constant(scale);
  } else {
    r.device(dev) = b / r.constant(scale);
    x.setZero();
  }
  p.device(dev) = r;
  float r_old = Norm2(r);

  for (Index icg = 0; icg < max_its; icg++) {
    q = op.A(p);
    float const alpha = r_old / Dot(p, q).real();
    x.device(dev) = x + p * p.constant(alpha);
    Log::Image(p, fmt::format(FMT_STRING("cg-p-{:02}"), icg));
    Log::Image(q, fmt::format(FMT_STRING("cg-q-{:02}"), icg));
    Log::Image(x, fmt::format(FMT_STRING("cg-x-{:02}"), icg));
    Log::Image(r, fmt::format(FMT_STRING("cg-r-{:02}"), icg));
    r.device(dev) = r - q * q.constant(alpha);
    float const r_new = Norm2(r);
    float const beta = r_new / r_old;
    p.device(dev) = r + p * p.constant(beta);
    float const nr = sqrt(r_new);
    Log::Print(FMT_STRING("CG {}: ɑ {} β {} norm resid {}"), icg, alpha, beta, nr);
    if (nr < thresh) {
      Log::Print(FMT_STRING("Reached convergence threshold"));
      break;
    }
    r_old = r_new;
  }
  return x * x.constant(scale);
}

/*
 * Pre-conditioned Conjugate gradients
 */
template <typename Op, typename Precond>
typename Op::Input pcg(
  Index const &max_its,
  float const &thresh,
  Op const &op,
  typename Op::Input const &b,
  typename Op::Input const &x0,
  Precond const *P)
{
  float const scale = Norm(b);
  Log::Print(
    FMT_STRING("Starting Preconditioned Conjugate Gradients, scale {} threshold {}"),
    scale,
    thresh);
  auto dev = Threads::GlobalDevice();
  // Allocate all memory
  using T = typename Op::Input;
  auto const dims = op.inputDimensions();
  CheckDims(dims, b.dimensions());
  CheckDims(dims, x0.dimensions());
  T q(dims), p(dims), r(dims), Pr(dims), x(dims);
  Log::Print("Initialising PCG with initial guess");
  Pr.device(dev) = (b - op.A(x0)) / r.constant(scale);
  r = P->apply(Pr);
  x.device(dev) = x0 / x.constant(scale);

  p.device(dev) = r;
  float r_old = std::real(Dot(r, Pr));

  for (Index icg = 0; icg < max_its; icg++) {
    q = op.A(p);
    float const alpha = r_old / Dot(p, q).real();
    x.device(dev) = x + p * p.constant(alpha);
    Log::Image(p, fmt::format(FMT_STRING("cg-p-{:02}"), icg));
    Log::Image(q, fmt::format(FMT_STRING("cg-q-{:02}"), icg));
    Log::Image(x, fmt::format(FMT_STRING("cg-x-{:02}"), icg));
    Log::Image(r, fmt::format(FMT_STRING("cg-r-{:02}"), icg));
    Pr.device(dev) = r - q * q.constant(alpha);
    r = P->apply(Pr);
    float const r_new = std::real(Dot(r, Pr));
    float const beta = r_new / r_old;
    p.device(dev) = r + p * p.constant(beta);
    float const nr = sqrt(r_new);
    Log::Print(FMT_STRING("CG {}: ɑ {} β {} norm resid {}"), icg, alpha, beta, nr);
    if (nr < thresh) {
      Log::Print(FMT_STRING("Reached convergence threshold"));
      break;
    }
    r_old = r_new;
  }
  return x * x.constant(scale);
}
