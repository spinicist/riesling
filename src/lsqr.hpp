#pragma once

#include "log.h"
#include "tensorOps.h"
#include "threads.h"
#include "types.h"

/* Based on https://github.com/PythonOptimizers/pykrylov/blob/master/pykrylov/lls/lsqr.py
 */
template <typename Op, typename Precond>
typename Op::Input lsqr(
  Index const &max_its,
  float const &thresh,
  Op &op,
  Precond const &pre,
  typename Op::Output const &b)
{
  Log::Print(FMT_STRING("Starting LSQR, threshold {}"), thresh);
  auto dev = Threads::GlobalDevice();
  // Allocate all memory
  using TI = typename Op::Input;
  using TO = typename Op::Output;
  auto const inDims = op.inputDimensions();
  auto const outDims = op.outputDimensions();
  TI x(inDims);
  x.setZero();

  TO Pu(outDims);
  TO u(outDims);
  Pu.device(dev) = b;
  u.device(dev) = pre->apply(Pu);
  float beta = sqrt(std::real(Dot(u, Pu)));
  Pu.device(dev) = Pu / b.constant(beta);
  u.device(dev) = u / b.constant(beta);

  TI v(inDims);
  v.device(dev) = op.Adj(u);
  float alpha = Norm(v);
  v.device(dev) = v / v.constant(alpha);

  TI w(inDims);
  w.device(dev) = v;

  float rho_ = alpha;
  float phi_ = beta;
  for (Index ii = 0; ii < max_its; ii++) {
    // Bidiagonalization step
    Pu.device(dev) = op.A(v) - alpha * Pu;
    u.device(dev) = pre->apply(Pu);
    beta = sqrt(std::real(Dot(Pu, u)));
    Pu.device(dev) = Pu / Pu.constant(beta);
    u.device(dev) = u / u.constant(beta);

    v.device(dev) = op.Adj(u) - beta * v;
    alpha = Norm(v);
    v.device(dev) = v / v.constant(alpha);

    // Apply orthogonal transformation
    float const rho = std::hypot(rho_, beta);
    float const c = rho_ / rho;
    float const s = beta / rho;
    float const theta = s * alpha;
    rho_ = -c * alpha;
    float const phi = c * phi_;
    phi_ = s * phi_;
    float const tau = s * phi;

    x.device(dev) = x + (phi / rho) * w;
    w.device(dev) = v - (theta / rho) * w;

    Log::Image(v, fmt::format(FMT_STRING("lsqr-v-{:02d}.nii"), ii));
    Log::Image(x, fmt::format(FMT_STRING("lsqr-x-{:02d}.nii"), ii));
    Log::Image(w, fmt::format(FMT_STRING("lsqr-w-{:02d}.nii"), ii));

    Log::Print(FMT_STRING("LSQR {}: ɑ {} β {} ɸ {} ρ {} θ {}"), ii, alpha, beta, phi, rho, theta);
    Log::Print(FMT_STRING("ρ_ {} c {} s {} ɸ_ {} τ {}"), rho_, c, s, phi_, tau);
  }
  return x;
}
