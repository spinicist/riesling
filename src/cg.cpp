#include "cg.h"
#include "threads.h"

void cg(SystemFunction const &sys, long const &max_its, float const &thresh, Cx3 &img, Log &log)
{
  log.info(FMT_STRING("Starting Conjugate Gradients, threshold {}"), thresh);

  // Allocate all memory
  auto const dims = img.dimensions();
  Cx3 bv(dims);
  Cx3 qv(dims);
  Cx3 pv(dims);
  Cx3 rv(dims);
  bv.setZero();
  qv.setZero();
  pv = img;
  rv = img;
  Eigen::Map<Eigen::VectorXcf> b(bv.data(), bv.size());
  Eigen::Map<Eigen::VectorXcf> q(qv.data(), qv.size());
  Eigen::Map<Eigen::VectorXcf> p(pv.data(), pv.size());
  Eigen::Map<Eigen::VectorXcf> r(rv.data(), rv.size());
  float r_old = r.dot(r).real();
  float const a2 = norm2(img);
  // auto const c = 6;
  for (long icg = 0; icg < max_its; icg++) {
    sys(pv, qv);
    float const alpha = r_old / p.dot(q).real();
    bv.device(Threads::GlobalDevice()) = bv + alpha * pv;
    rv.device(Threads::GlobalDevice()) = rv - alpha * qv;
    float const r_new = r.dot(r).real();
    float const beta = r_new / r_old;
    pv.device(Threads::GlobalDevice()) = rv + beta * pv;
    float const delta = r_new / a2;
    log.info(FMT_STRING("Finished CG-Step {}: Alpha {} Beta {} Delta {}"), icg, alpha, beta, delta);
    if (delta < thresh) {
      break;
    }
    r_old = r_new;
  }
  img = bv;
}
