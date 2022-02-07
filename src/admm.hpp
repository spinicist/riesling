#pragma once

#include "cg.hpp"
#include "precond/none.hpp"
#include "tensorOps.h"
#include "threads.h"

template <typename Op>
struct AugmentedOp
{
  using Input = typename Op::Input;
  Op const &op;
  float rho;

  auto inputDimensions() const
  {
    return op.inputDimensions();
  }

  Input A(typename Op::Input const &x) const
  {
    return Input(op.AdjA(x) + rho * x);
  }
};

template <typename Op, typename Precond>
typename Op::Input admm(
  Index const outer_its,
  Index const lsq_its,
  float const lsq_thresh,
  Op const &op,
  Precond const &pre,
  std::function<Cx4(Cx4 const &)> const &reg,
  float const rho,
  typename Op::Output const &b)
{
  Log::Print(FMT_STRING("Starting ADMM"));
  auto dev = Threads::GlobalDevice();
  // Allocate all memory
  using T = typename Op::Input;
  auto const dims = op.inputDimensions();
  T x0(dims);
  x0.device(dev) = op.Adj(pre(b));
  T x(dims);
  T z(dims);
  T u(dims);
  T xpu(dims);
  x.setZero();
  z.setZero();
  u.setZero();
  xpu.setZero();

  // Augment system
  AugmentedOp<Op> augmented{op, rho};

  for (Index ii = 0; ii < outer_its; ii++) {
    x.device(dev) = x0 + x0.constant(rho) * (z - u);
    x = cg(lsq_its, lsq_thresh, augmented, NoPrecond<typename Op::Input>(), x0, x);
    xpu.device(dev) = x + u;
    z = reg(xpu);
    u.device(dev) = xpu - z;
    Log::Print(FMT_STRING("Finished ADMM iteration {}"), ii);
    Log::Image(x, fmt::format("admm-x-{:02d}.nii", ii));
    Log::Image(xpu, fmt::format("admm-xpu-{:02d}.nii", ii));
    Log::Image(z, fmt::format("admm-z-{:02d}.nii", ii));
    Log::Image(u, fmt::format("admm-u-{:02d}.nii", ii));
  }
  return x;
}
