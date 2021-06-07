#include "cg.h"

#include "tensorOps.h"
#include "threads.h"

void cg(CgSystem const &sys, long const &max_its, float const &thresh, Cx3 &img, Log &log)
{
  log.info(FMT_STRING("Starting Conjugate Gradients, threshold {}"), thresh);

  // Allocate all memory
  auto const dims = img.dimensions();
  Cx3 b(dims);
  Cx3 q(dims);
  Cx3 p(dims);
  Cx3 r(dims);
  b.setZero();
  q.setZero();
  p = img;
  r = img;
  float r_old = Norm2(r);
  float const a2 = Norm2(img);

  for (long icg = 0; icg < max_its; icg++) {
    sys(p, q);
    float const alpha = r_old / std::real(Dot(p, q));
    b.device(Threads::GlobalDevice()) = b + p * p.constant(alpha);
    r.device(Threads::GlobalDevice()) = r - q * q.constant(alpha);
    float const r_new = Norm2(r);
    float const beta = r_new / r_old;
    p.device(Threads::GlobalDevice()) = r + p * p.constant(beta);
    float const delta = r_new / a2;
    log.image(b, fmt::format(FMT_STRING("cg-b-{:02}.nii"), icg));
    log.info(FMT_STRING("CG {}: ɑ {} β {} δ {}"), icg, alpha, beta, delta);
    if (delta < thresh) {
      log.info(FMT_STRING("Reached convergence threshold"));
      break;
    }
    r_old = r_new;
  }
  img = b;
}

void cgvar(
    CgVarSystem const &sys,
    long const &max_its,
    float const &thresh,
    float const &pre0,
    float const &pre1,
    Cx3 &img,
    Log &log)
{
  log.info(FMT_STRING("Starting Variably Preconditioned Conjugate Gradients"));

  // Allocate all memory
  auto const dims = img.dimensions();
  Cx3 b(dims);
  Cx3 q(dims);
  Cx3 p(dims);
  Cx3 r(dims);
  Cx3 r1(dims);
  b.setZero();
  q.setZero();
  p = img;
  r = img;
  float const a2 = Norm2(img);

  for (long icg = 0; icg < max_its; icg++) {
    float const pre = pre0 + (pre1 - pre0) * (1.f * icg) / (max_its - 1.f);
    sys(p, q, pre);
    r1 = r;
    float const r_old = Norm2(r1);
    float const alpha = r_old / std::real(Dot(p, q));
    b.device(Threads::GlobalDevice()) = b + p * p.constant(alpha);

    r.device(Threads::GlobalDevice()) = r - q * q.constant(alpha);
    float const r_new = std::real(Dot(r, r - r1));
    float const beta = r_new / r_old;
    p.device(Threads::GlobalDevice()) = r + p * p.constant(beta);
    float const delta = r_new / a2;
    log.image(b, fmt::format(FMT_STRING("cg-b-{:02}.nii"), icg));
    log.info(FMT_STRING("CG {}: ɑ {} β {} δ {} pre {}"), icg, alpha, beta, delta, pre);
    if (delta < thresh) {
      log.info(FMT_STRING("Reached convergence threshold"));
      break;
    }
  }
  img = b;
}