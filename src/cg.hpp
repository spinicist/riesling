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
void cg(long const &max_its, float const &thresh, Op const &op, typename Op::Input &x, Log &log)
{
  log.info(FMT_STRING("Starting Conjugate Gradients, threshold {}"), thresh);
  auto dev = Threads::GlobalDevice();
  float const norm_x0 = Norm2(x);
  // Allocate all memory
  using T = typename Op::Input;
  auto const dims = x.dimensions();
  T q(dims);
  T p(dims);
  T r(dims);
  q.setZero();
  op.AdjA(x, r);
  r.device(dev) = x - r;
  p.device(dev) = r;
  float r_old = Norm2(r);

  for (long icg = 0; icg < max_its; icg++) {
    op.AdjA(p, q);
    float const alpha = r_old / std::real(Dot(p, q));
    x.device(dev) = x + p * p.constant(alpha);
    r.device(dev) = r - q * q.constant(alpha);
    float const r_new = Norm2(r);
    float const beta = r_new / r_old;
    p.device(dev) = r + p * p.constant(beta);
    float const delta = r_new / norm_x0;
    log.image(x, fmt::format(FMT_STRING("cg-b-{:02}.nii"), icg));
    log.info(FMT_STRING("CG {}: ɑ {} β {} δ {}"), icg, alpha, beta, delta);
    if (delta < thresh) {
      log.info(FMT_STRING("Reached convergence threshold"));
      break;
    }
    r_old = r_new;
  }
}

template <typename Op>
void cgvar(
    long const &max_its,
    float const &thresh,
    float const &pre0,
    float const &pre1,
    Op &op,
    typename Op::Input &x,
    Log &log)
{
  log.info(FMT_STRING("Starting Variably Preconditioned Conjugate Gradients"));
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

  for (long icg = 0; icg < max_its; icg++) {
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
    log.image(b, fmt::format(FMT_STRING("cg-b-{:02}.nii"), icg));
    log.info(FMT_STRING("CG {}: ɑ {} β {} δ {} pre {}"), icg, alpha, beta, delta, pre);
    if (delta < thresh) {
      log.info(FMT_STRING("Reached convergence threshold"));
      break;
    }
  }
  x = b;
}
