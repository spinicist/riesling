#pragma once

#include "log.h"
#include "tensorOps.h"
#include "threads.h"
#include "types.h"

/*
 * Conjugate gradients on the normal equations.
 */
template <typename Op, typename Precond>
typename Op::Input cgnorm(
  Index const &max_its,
  float const &thresh,
  Op const &op,
  Precond const &pre,
  typename Op::Output const &b,
  typename Op::Input const &x0 = typename Op::Input())
{
  Log::Print(FMT_STRING("Starting Conjugate Gradients, threshold {}"), thresh);
  auto dev = Threads::GlobalDevice();
  // Allocate all memory
  using T = typename Op::Input;
  auto const dims = op.inputDimensions();
  T q(dims);
  T p(dims);
  T r(dims);
  T x(dims);
  r.device(dev) = op.Adj(pre(b));
  // If we have an initial guess, use it
  if (x0.size()) {
    if (x0.dimensions() != dims) {
      Log::Fail(
        FMT_STRING("Operator dims {} not equal to x0 dims {}"),
        fmt::join(dims, ","),
        fmt::join(x0.dimensions(), ","));
    }
    r.device(dev) = r - op.Adj(pre(op.A(x0)));
    x.device(dev) = x0;
  } else {
    x.setZero();
  }
  p.device(dev) = r;

  float r_old = Norm2(r);
  float const n0 = sqrt(r_old);

  for (Index icg = 0; icg < max_its; icg++) {
    q = op.Adj(pre(op.A(p)));
    Cx const pdq = Dot(p, q);
    Log::Debug(FMT_STRING("p.q = {}"), pdq);
    float const alpha = r_old / std::real(pdq);
    x.device(dev) = x + p * p.constant(alpha);
    Log::Image(p, fmt::format(FMT_STRING("cg-p-{:02}.nii"), icg));
    Log::Image(q, fmt::format(FMT_STRING("cg-q-{:02}.nii"), icg));
    Log::Image(x, fmt::format(FMT_STRING("cg-x-{:02}.nii"), icg));
    Log::Image(r, fmt::format(FMT_STRING("cg-r-{:02}.nii"), icg));
    r.device(dev) = r - q * q.constant(alpha);
    float const r_new = Norm2(r);
    float const beta = r_new / r_old;
    p.device(dev) = r + p * p.constant(beta);
    float const ni = sqrt(r_new) / n0;
    Log::Print(FMT_STRING("CG {}: ɑ {} β {} norm resid {}"), icg, alpha, beta, ni);
    if (ni < thresh) {
      Log::Print(FMT_STRING("Reached convergence threshold"));
      break;
    }
    r_old = r_new;
  }
  return x;
}

/*
 * Conjugate gradients not on the normal equations (for ADMM)
 */
template <typename Op, typename Precond>
typename Op::Input cg(
  Index const &max_its,
  float const &thresh,
  Op const &op,
  Precond const &pre,
  typename Op::Input const &b,
  typename Op::Input &x0 = typename Op::Input())
{
  Log::Print(FMT_STRING("Starting Conjugate Gradients, threshold {}"), thresh);
  auto dev = Threads::GlobalDevice();
  // Allocate all memory
  using T = typename Op::Input;
  auto const dims = op.inputDimensions();
  T q(dims);
  T p(dims);
  T r(dims);
  T x(dims);
  r.device(dev) = pre(b);
  // If we have an initial guess, use it
  if (x0.size()) {
    if (x0.dimensions() != dims) {
      Log::Fail(
        FMT_STRING("Operator dims {} not equal to x0 dims {}"),
        fmt::join(dims, ","),
        fmt::join(x0.dimensions(), ","));
    }
    r.device(dev) = r - op.A(x0);
    x.device(dev) = x0;
  } else {
    x.setZero();
  }
  p.device(dev) = r;
  float r_old = Norm2(r);
  float const n0 = sqrt(r_old);

  for (Index icg = 0; icg < max_its; icg++) {
    q = op.A(p);
    Cx const pdq = Dot(p, q);
    Log::Debug(FMT_STRING("p.q = {}"), pdq);
    float const alpha = r_old / std::real(pdq);
    x.device(dev) = x + p * p.constant(alpha);
    Log::Image(p, fmt::format(FMT_STRING("cg-p-{:02}.nii"), icg));
    Log::Image(q, fmt::format(FMT_STRING("cg-q-{:02}.nii"), icg));
    Log::Image(x, fmt::format(FMT_STRING("cg-x-{:02}.nii"), icg));
    Log::Image(r, fmt::format(FMT_STRING("cg-r-{:02}.nii"), icg));
    r.device(dev) = r - q * q.constant(alpha);
    float const r_new = Norm2(r);
    float const beta = r_new / r_old;
    p.device(dev) = r + p * p.constant(beta);
    float const ni = sqrt(r_new) / n0;
    Log::Print(FMT_STRING("CG {}: ɑ {} β {} norm resid {}"), icg, alpha, beta, ni);
    if (ni < thresh) {
      Log::Print(FMT_STRING("Reached convergence threshold"));
      break;
    }
    r_old = r_new;
  }
  return x;
}
