#pragma once

#include "log.h"
#include "tensorOps.h"
#include "threads.h"
#include "types.h"

/* Conjugate gradients as described in K. P. Pruessmann, M. Weiger, M. B. Scheidegger, and P.
 * Boesiger, ‘SENSE: Sensitivity encoding for fast MRI’, Magnetic Resonance in Medicine, vol. 42,
 * no. 5, pp. 952–962, 1999.
 */
template <typename Op>
void cg(Index const &max_its, float const &thresh, Op const &op, typename Op::Input &x)
{
  Log::Print(FMT_STRING("Starting Conjugate Gradients, threshold {}"), thresh);
  auto dev = Threads::GlobalDevice();
  // Allocate all memory
  using T = typename Op::Input;
  auto const dims = x.dimensions();
  T q(dims);
  T p(dims);
  T r(dims);
  r.device(dev) = x;
  x.setZero();
  p.device(dev) = r;

  float r_old = Norm2(r);
  float const n0 = sqrt(r_old);

  for (Index icg = 0; icg < max_its; icg++) {
    q = op.AdjA(p);
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
}

template <typename Op>
void cgvar(
  Index const &max_its,
  float const &thresh,
  float const &pre0,
  float const &pre1,
  Op &op,
  typename Op::Input &x)
{
  Log::Print(FMT_STRING("Starting Variably Preconditioned Conjugate Gradients"));
  auto dev = Threads::GlobalDevice();
  // Allocate all memory
  using T = typename Op::Input;
  auto const dims = x.dimensions();
  T b(dims);
  T q(dims);
  T p(dims);
  T r(dims);
  T r1(dims);
  b.setZero();
  q.setZero();
  p = x;
  r = x;
  float const a2 = Norm2(x);

  for (Index icg = 0; icg < max_its; icg++) {
    float const prog = static_cast<float>(icg) / ((max_its == 1) ? 1. : (max_its - 1.f));
    float const pre = std::exp(std::log(pre1) * prog + std::log(pre0) * (1.f - prog));
    op.setPreconditioning(pre);
    op.AdjA(p, q);
    r1 = r;
    float const r_old = Norm2(r1);
    float const alpha = r_old / std::real(Dot(p, q));
    b.device(dev) = b + p * p.constant(alpha);

    r.device(dev) = r - q * q.constant(alpha);
    float const r_new = std::real(Dot(r, r - r1));
    float const beta = r_new / r_old;
    p.device(dev) = r + p * p.constant(beta);
    float const delta = r_new / a2;
    Log::Image(b, fmt::format(FMT_STRING("cg-b-{:02}.nii"), icg));
    Log::Print(FMT_STRING("CG {}: ɑ {} β {} δ {} pre {}"), icg, alpha, beta, delta, pre);
    if (delta < thresh) {
      Log::Print(FMT_STRING("Reached convergence threshold"));
      break;
    }
  }
  x = b;
}
