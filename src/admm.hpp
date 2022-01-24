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

  auto AdjA(typename Op::Input const &x) const
  {
    return op.AdjA(x) + rho * x;
  }
};

template <typename Op>
void admm(
  Index const outer_its,
  Index const lsq_its,
  float const lsq_thresh,
  Op const &lsq_op,
  float const rho,
  std::function<Cx4(Cx4 const &)> const &reg,
  typename Op::Input &x)
{
  if (outer_its < 1)
    return;
  Log::Print(FMT_STRING("Starting ADMM"));
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

  for (Index ii = 0; ii < outer_its; ii++) {
    x.device(dev) = b + b.constant(rho) * (z - u);
    cg(lsq_its, lsq_thresh, augmented, x);
    xpu.device(dev) = x + u;
    z = reg(xpu);
    u.device(dev) = xpu - z;
    Log::Print("Finished ADMM iteration {}", ii);
    Log::Image(x, fmt::format("admm-x-{:02d}.nii", ii));
    Log::Image(xpu, fmt::format("admm-xpu-{:02d}.nii", ii));
    Log::Image(z, fmt::format("admm-z-{:02d}.nii", ii));
    Log::Image(u, fmt::format("admm-u-{:02d}.nii", ii));
  }
}
