#include "cg.h"

#include "tensorOps.h"
#include "threads.h"

void cg(SystemFunction const &sys, long const &max_its, float const &thresh, Cx3 &img, Log &log)
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
      break;
    }
    r_old = r_new;
  }
  img = b;
}
