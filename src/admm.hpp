#pragma once

#include "cg.hpp"
#include "tensorOps.h"
#include "threads.h"

template <typename Op>
struct AugmentedOp
{
  using Input = typename Op::Input;
  Op const &op;
  float rho;

  void AdjA(typename Op::Input const &x, typename Op::Input &y) const
  {
    op.AdjA(x, y);
    y.device(Threads::GlobalDevice()) = y + rho * x;
  }
};

template <typename Op>
void admm(
  long const outer_its,
  long const lsq_its,
  float const lsq_thresh,
  Op const &lsq_op,
  float const rho,
  std::function<Cx4(Cx4 const &)> const &reg,
  typename Op::Input &x,
  Log &log)
{
  if (outer_its < 1)
    return;
  log.info(FMT_STRING("Starting ADMM"));
  auto dev = Threads::GlobalDevice();
  // Allocate all memory
  using T = typename Op::Input;
  auto const dims = x.dimensions();
  T b(dims);
  T z(dims);
  T u(dims);
  T xpu(dims);
  b = x;
  z.setZero();
  u.setZero();
  xpu.setZero();

  // Augment system
  AugmentedOp<Op> augmented{lsq_op, rho};

  for (long ii = 0; ii < outer_its; ii++) {
    x.device(dev) = b + b.constant(rho) * (z - u);
    cg(lsq_its, lsq_thresh, augmented, x, log);
    xpu.device(dev) = x + u;
    z = reg(xpu);
    u.device(dev) = xpu - z;
    log.info("Finished ADMM iteration {}", ii);
    log.image(x, fmt::format("admm-x-{:02d}.nii", ii));
    log.image(xpu, fmt::format("admm-xpu-{:02d}.nii", ii));
    log.image(z, fmt::format("admm-z-{:02d}.nii", ii));
    log.image(u, fmt::format("admm-u-{:02d}.nii", ii));
  }
}
