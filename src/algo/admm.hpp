#pragma once

#include "cg.hpp"
#include "lsmr.hpp"
#include "lsqr.hpp"
#include "tensorOps.h"
#include "threads.h"

template <typename Op, typename LeftPrecond>
typename Op::Input admm_lsmr(
  Index const outer_its,
  float rho,
  std::function<Cx4(Cx4 const &)> const &reg,
  Index const lsq_its,
  Op const &op,
  typename Op::Output const &b,
  LeftPrecond const *M = nullptr, // Left preconditioner
  float const atol = 1.e-6f,
  float const btol = 1.e-6f,
  float const ctol = 1.e-6f,
  float const abstol = 1.e-3f,
  float const reltol = 1.e-3f)
{
  Log::Print(FMT_STRING("ADMM LSMR rho {}"), rho);
  auto dev = Threads::GlobalDevice();
  // Allocate all memory
  using T = typename Op::Input;
  auto const dims = op.inputDimensions();
  Index const nvox = Product(dims);
  T u(dims), x(dims), z(dims), zold(dims), xpu(dims);
  x.setZero();
  z.setZero();
  zold.setZero();
  u.setZero();
  xpu.setZero();

  for (Index ii = 0; ii < outer_its; ii++) {
    x = lsmr(lsq_its, op, b, atol, btol, ctol, rho, M, x, (z - u), (ii == 2));

    xpu.device(dev) = x + u;
    zold = z;
    z = reg(xpu);
    u.device(dev) = xpu - z;

    float const norm_prim = Norm(x - z);
    float const norm_dual = Norm(-rho * (z - zold));

    float const eps_prim = sqrtf(nvox) * abstol + reltol * std::max(Norm(x), Norm(z));
    float const eps_dual = sqrtf(nvox) * abstol + reltol * rho * Norm(u);

    Log::Tensor(x, fmt::format("admm-x-{:02d}", ii));
    Log::Tensor(xpu, fmt::format("admm-xpu-{:02d}", ii));
    Log::Tensor(z, fmt::format("admm-z-{:02d}", ii));
    Log::Tensor(u, fmt::format("admm-u-{:02d}", ii));

    Log::Print(
      FMT_STRING("ADMM {:02d}: Primal Norm {} Primal Eps {} Dual Norm {} Dual Eps {}"),
      ii,
      norm_prim,
      eps_prim,
      norm_dual,
      eps_dual);
    if ((norm_prim < eps_prim) && (norm_dual < eps_dual)) {
      break;
    }
    float const mu = 10.f;
    float const tau = 2.f;
    if (norm_prim > mu * norm_dual) {
      rho = rho * tau;
      u.device(dev) = u / u.constant(tau);
      Log::Print(FMT_STRING("Rescaled rho to {}"), rho);
    } else if (norm_dual > mu * norm_prim) {
      rho = rho / tau;
      u.device(dev) = u * u.constant(tau);
      Log::Print(FMT_STRING("Rescaled rho to {}"), rho);
    }
  }
  return x;
}

template <typename Op, typename LeftPrecond>
typename Op::Input admm_lsqr(
  Index const outer_its,
  float rho,
  std::function<typename Op::Input(typename Op::Input const &)> const &reg,
  Index const lsq_its,
  Op const &op,
  typename Op::Output const &b,
  LeftPrecond const *M = nullptr, // Left preconditioner
  float const atol = 1.e-6f,
  float const btol = 1.e-6f,
  float const ctol = 1.e-6f,
  float const abstol = 1.e-3f,
  float const reltol = 1.e-3f)
{
  auto dev = Threads::GlobalDevice();
  // Allocate all memory
  using T = typename Op::Input;
  auto const dims = op.inputDimensions();
  Index const nvox = Product(dims);
  T u(dims), x(dims), z(dims), zold(dims), xpu(dims);
  x.setZero();
  z.setZero();
  zold.setZero();
  u.setZero();
  xpu.setZero();

  Log::Print(FMT_STRING("ADMM LSQR rho {}"), rho);
  for (Index ii = 0; ii < outer_its; ii++) {
    x = lsqr(lsq_its, op, b, atol, btol, ctol, rho, M, (ii == 1), x, (z - u));
    xpu.device(dev) = x + u;
    zold = z;
    z = reg(xpu);
    u.device(dev) = xpu - z;

    float const norm_prim = Norm(x - z);
    float const norm_dual = Norm(-rho * (z - zold));

    float const eps_prim = sqrtf(nvox) * abstol + reltol * std::max(Norm(x), Norm(-z));
    float const eps_dual = sqrtf(nvox) * abstol + reltol * rho * Norm(u);

    Log::Tensor(x, fmt::format("admm-x-{:02d}", ii));
    Log::Tensor(xpu, fmt::format("admm-xpu-{:02d}", ii));
    Log::Tensor(z, fmt::format("admm-z-{:02d}", ii));
    Log::Tensor(u, fmt::format("admm-u-{:02d}", ii));

    Log::Print(
      FMT_STRING("ADMM {:02d}: Primal Norm {} Primal Eps {} Dual Norm {} Dual Eps {}"),
      ii,
      norm_prim,
      eps_prim,
      norm_dual,
      eps_dual);
    if ((norm_prim < eps_prim) && (norm_dual < eps_dual)) {
      break;
    }
    float const mu = 10.f;
    float const tau = 2.f;
    if (norm_prim > mu * norm_dual) {
      rho = rho * tau;
      u.device(dev) = u / u.constant(tau);
      Log::Print(FMT_STRING("Rescaled rho to {}"), rho);
    } else if (norm_dual > mu * norm_prim) {
      rho = rho / tau;
      u.device(dev) = u * u.constant(tau);
      Log::Print(FMT_STRING("Rescaled rho to {}"), rho);
    }
  }
  return x;
}

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

  auto outputDimensions() const
  {
    return op.inputDimensions();
  }

  Input A(typename Op::Input const &x) const
  {
    return Input(op.AdjA(x) + rho * x);
  }
};

template <typename Op>
typename Op::Input admm_cg(
  Index const outer_its,
  Index const lsq_its,
  float const lsq_thresh,
  Op const &op,
  std::function<Cx4(Cx4 const &)> const &reg,
  float rho,
  typename Op::Output const &b,
  float const abstol = 1.e-3f,
  float const reltol = 1.e-3f)
{
  Log::Print(FMT_STRING("ADMM-CG rho {}"), rho);
  auto dev = Threads::GlobalDevice();
  // Allocate all memory
  using T = typename Op::Input;
  auto const dims = op.inputDimensions();
  Index const N = Product(dims);
  T x0(dims), x(dims), z(dims), zold(dims), u(dims), xpu(dims);
  x0.device(dev) = op.Adj(b);
  x.setZero();
  z.setZero();
  u.setZero();
  xpu.setZero();

  // Augment system
  AugmentedOp<Op> augmented{op, rho};
  for (Index ii = 0; ii < outer_its; ii++) {
    x = cg(lsq_its, lsq_thresh, augmented, x0 + x0.constant(rho) * (z - u), x);
    xpu.device(dev) = x + u;
    zold = z;
    z = reg(xpu);
    u.device(dev) = xpu - z;

    float const norm_prim = Norm(x - z);
    float const norm_dual = Norm(-rho * (z - zold));

    float const eps_prim = sqrtf(N) * abstol + reltol * std::max(Norm(x), Norm(-z));
    float const eps_dual = sqrtf(N) * abstol + reltol * rho * Norm(u);

    Log::Tensor(x, fmt::format("admm-x-{:02d}", ii));
    Log::Tensor(xpu, fmt::format("admm-xpu-{:02d}", ii));
    Log::Tensor(z, fmt::format("admm-z-{:02d}", ii));
    Log::Tensor(u, fmt::format("admm-u-{:02d}", ii));

    Log::Print(
      FMT_STRING("ADMM-CG {:02d}: Primal Norm {} Primal Eps {} Dual Norm {} Dual Eps {}"),
      ii,
      norm_prim,
      eps_prim,
      norm_dual,
      eps_dual);
    if ((norm_prim < eps_prim) && (norm_dual < eps_dual)) {
      break;
    }
    float const mu = 10.f;
    float const tau = 2.f;
    if (norm_prim > mu * norm_dual) {
      rho = rho * tau;
      u.device(dev) = u / u.constant(tau);
      Log::Print(FMT_STRING("Rescaled rho to {}"), rho);
    } else if (norm_dual > mu * norm_prim) {
      rho = rho / tau;
      u.device(dev) = u * u.constant(tau);
      Log::Print(FMT_STRING("Rescaled rho to {}"), rho);
    }
  }
  return x;
}