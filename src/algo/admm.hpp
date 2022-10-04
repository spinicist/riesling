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
  Functor<Input> *reg;
  Index iterLimit;
  float ρ = 0.1; // Langrangian
  float α = 1.f; // Over-relaxation
  float abstol = 1.e-3f;
  float reltol = 1.e-3f;

  Input run(Output const &b) const
  {
    auto dev = Threads::GlobalDevice();
    // Allocate all memory
    auto const dims = inner.op.inputDimensions();
    Input u(dims), x(dims), z(dims), zold(dims), xpu(dims);
    x.setZero();
    z.setZero();
    zold.setZero();
    u.setZero();
    xpu.setZero();

    Log::Print(FMT_STRING("ADMM ρ {}"), ρ);
    float const absThresh = abstol * std::sqrt(float(Product(dims)));
    for (Index ii = 0; ii < iterLimit; ii++) {
      x = inner.run(b, x, (z - u));
      xpu.device(dev) = x * x.constant(α) + z * z.constant(1 - α) + u;
      zold = z;
      z = (*reg)(xpu);
      u.device(dev) = xpu - z;

      float const pNorm = Norm(x - z);
      float const dNorm = Norm(-ρ * (z - zold));

      float const normx = Norm(x);
      float const normz = Norm(z);
      float const normu = Norm(u);

      float const pEps = absThresh + reltol * std::max(normx, normz);
      float const dEps = absThresh + reltol * ρ * normu;

      Log::Tensor(x, fmt::format("admm-x-{:02d}", ii));
      Log::Tensor(z, fmt::format("admm-z-{:02d}", ii));
      Log::Tensor(u, fmt::format("admm-u-{:02d}", ii));
      Log::Print(
        FMT_STRING("ADMM {:02d}: Primal || {} ε {} Dual || {} ε {} |u| {} |x| {} |z| {}"),
        ii,
        pNorm,
        pEps,
        dNorm,
        dEps,
        normu,
        normx,
        normz);
      if ((pNorm < pEps) && (dNorm < dEps)) {
        break;
      }
      float const mu = 10.f;
      if (pNorm > mu * dNorm) {
        Log::Print(FMT_STRING("Primal norm is outside limit {}, consider increasing ρ"), mu * dNorm);
      } else if (dNorm > mu * pNorm) {
        Log::Print(FMT_STRING("Dual norm is outside limit {}, consider decreasing ρ"), mu * pNorm);
      }
    }
    return x;
  }
};

} // namespace rl
