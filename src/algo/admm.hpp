#pragma once

#include "func/functor.hpp"
#include "log.hpp"
#include "tensorOps.hpp"
#include "threads.hpp"

namespace rl {

template <typename Inner>
struct ADMM
{
  using Input = typename Inner::Input;
  using Output = typename Inner::Output;

  Inner &inner;
  std::shared_ptr<Prox<Input>> prox;
  Index iterLimit = 8;
  float λ = 0.;  // Proximal operator parameter
  float α = 1.f; // Over-relaxation
  float μ = 10.f; // Primal-dual mismatch limit
  float τ = 2.f;  // Primal-dual mismatch rescale
  float abstol = 1.e-3f;
  float reltol = 1.e-3f;

  Input run(Eigen::TensorMap<Output const> b, float ρ) const
  {
    auto dev = Threads::GlobalDevice();
    // Allocate all memory
    auto const dims = inner.op->inputDimensions();
    Input u(dims), x(dims), z(dims), zold(dims), xpu(dims);

    // Set initial values
    x.setZero();
    z.setZero();
    u.setZero();

    float const absThresh = abstol * Norm(x);
    Log::Print(FMT_STRING("ADMM ρ {} Abs Thresh {}"), ρ, absThresh);
    for (Index ii = 0; ii < iterLimit; ii++) {
      x = inner.run(b, ρ, x, (z - u));
      xpu.device(dev) = x * x.constant(α) + z * z.constant(1 - α) + u;
      zold = z;
      z = (*prox)(λ / ρ, xpu);
      u.device(dev) = xpu - z;

      float const pNorm = Norm(x - z);
      float const dNorm = ρ * Norm(z - zold);

      float const normx = Norm(x);
      float const normz = Norm(z);
      float const normu = Norm(u);

      float const pEps = absThresh + reltol * std::max(normx, normz);
      float const dEps = absThresh + reltol * ρ * normu;

      Log::Tensor(x, fmt::format("admm-x-{:02d}", ii));
      Log::Tensor(z, fmt::format("admm-z-{:02d}", ii));
      Log::Tensor(u, fmt::format("admm-u-{:02d}", ii));
      Log::Print(
        FMT_STRING("ADMM {:02d}: Primal || {} ε {} Dual || {} ε {} |x| {} |z| {} |u| {}"),
        ii,
        pNorm,
        pEps,
        dNorm,
        dEps,
        normx,
        normz,
        normu);
      if ((pNorm < pEps) && (dNorm < dEps)) {
        break;
      }
      if (pNorm > μ * dNorm) {
        ρ *= τ;
        Log::Print(FMT_STRING("Primal norm outside limit {}, rescaled ρ to {}"), μ * dNorm, ρ);
      } else if (dNorm > μ * pNorm) {
        ρ /= τ;
        Log::Print(FMT_STRING("Dual norm outside limit {}, rescaled ρ to "), μ * pNorm, ρ);
      }
    }
    return x;
  }
};

} // namespace rl
