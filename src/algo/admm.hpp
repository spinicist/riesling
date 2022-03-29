#pragma once

#include "lsmr.hpp"
#include "tensorOps.h"
#include "threads.h"

template <typename Op, typename Precond>
typename Op::Input admm(
  Index const outer_its,
  float const rho,
  std::function<Cx4(Cx4 const &)> const &reg,
  Index const lsq_its,
  Op const &op,
  typename Op::Output const &b,
  Precond const *M = nullptr, // Left preconditioner
  float const atol = 1.e-6f,
  float const btol = 1.e-6f,
  float const ctol = 1.e-6f)
{
  Log::Print(FMT_STRING("Starting ADMM"));
  auto dev = Threads::GlobalDevice();
  // Allocate all memory
  using T = typename Op::Input;
  auto const dims = op.inputDimensions();
  T x0(dims);
  x0.device(dev) = op.Adj(b);
  T u(dims), x(dims), z(dims), xpu(dims);
  x.setZero();
  z.setZero();
  u.setZero();
  xpu.setZero();

  for (Index ii = 0; ii < outer_its; ii++) {
    x = lsmr(lsq_its, op, b, M, atol, btol, ctol, rho, (z - u));
    xpu.device(dev) = x + u;
    z = reg(xpu);
    u.device(dev) = xpu - z;
    Log::Print(FMT_STRING("Finished ADMM iteration {}"), ii);
    Log::Image(x, fmt::format("admm-x-{:02d}", ii));
    Log::Image(xpu, fmt::format("admm-xpu-{:02d}", ii));
    Log::Image(z, fmt::format("admm-z-{:02d}", ii));
    Log::Image(u, fmt::format("admm-u-{:02d}", ii));
  }
  return x;
}
