#pragma once

#include "common.hpp"
#include "log.h"
#include "tensorOps.h"
#include "threads.h"
#include "types.h"

namespace rl {
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

  auto outputDimensions() const
  {
    return op.inputDimensions();
  }

  Input A(typename Op::Input const &x) const
  {
    return Input(op.AdjA(x));
  }
};

/*
 * Conjugate gradients
 */
template <typename Op>
typename Op::Input cg(
  Index const &max_its,
  float const &tol,
  Op const &op,
  typename Op::Input const &b,
  typename Op::Input const &x0 = typename Op::Input(),
  bool const debug = false)
{
  auto dev = Threads::GlobalDevice();
  // Allocate all memory
  using T = typename Op::Input;
  CheckDimsEqual(op.outputDimensions(), b.dimensions());
  auto const dims = op.inputDimensions();
  T q(dims), p(dims), r(dims), x(dims);
  // If we have an initial guess, use it
  if (x0.size()) {
    CheckDimsEqual(dims, x0.dimensions());
    Log::Print("Warm-start CG");
    r.device(dev) = b - op.A(x0);
    x.device(dev) = x0;
  } else {
    r.device(dev) = b;
    x.setZero();
  }
  p.device(dev) = r;
  float r_old = Norm2(r);
  float const thresh = tol * sqrt(r_old);
  Log::Print(FMT_STRING("CG    |r| {:5.3E} threshold {:5.3E}"), sqrt(r_old), thresh);
  for (Index icg = 0; icg < max_its; icg++) {
    q = op.A(p);
    float const alpha = r_old / Dot(p, q).real();
    x.device(dev) = x + p * p.constant(alpha);
    if (debug) {
      Log::Tensor(p, fmt::format(FMT_STRING("cg-p-{:02}"), icg));
      Log::Tensor(q, fmt::format(FMT_STRING("cg-q-{:02}"), icg));
      Log::Tensor(x, fmt::format(FMT_STRING("cg-x-{:02}"), icg));
      Log::Tensor(r, fmt::format(FMT_STRING("cg-r-{:02}"), icg));
    }
    r.device(dev) = r - q * q.constant(alpha);
    float const r_new = Norm2(r);
    float const beta = r_new / r_old;
    p.device(dev) = r + p * p.constant(beta);
    float const nr = sqrt(r_new);
    Log::Print(FMT_STRING("CG {:02d} |r| {:5.3E} ɑ {:5.3E} β {:5.3E} |x| {:5.3E}"), icg, nr, alpha, beta, Norm(x));
    if (nr < thresh) {
      Log::Print(FMT_STRING("Reached convergence threshold"));
      break;
    }
    r_old = r_new;
  }
  return x;
}
}
