#pragma once

#include "cg.hpp"
#include "func/functor.hpp"
#include "log.hpp"
#include "tensorOps.hpp"
#include "threads.hpp"

namespace rl {

template <typename Op>
struct AugmentedOp
{
  using Input = typename Op::Input;
  using InputMap = typename Op::InputMap;
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

  auto forward(Input const &x) const -> Input
  {
    Input xcopy = x;
    xcopy = op.adjfwd(xcopy) + rho * x;
    return xcopy;
  }
};

template <typename Inner>
struct AugmentedADMM
{
  using Input = typename Inner::Input;

  Inner &inner;
  Prox<Input> *reg;
  Index iterLimit;
  float λ = 0.;  // Proximal operator parameter
  float ρ = 0.1; // Langrangian
  float abstol = 1.e-3f;
  float reltol = 1.e-3f;

  Input run(Input const &x0) const
  {
    Log::Print(FMT_STRING("ADMM-CG rho {}"), ρ);
    auto dev = Threads::GlobalDevice();
    // Allocate all memory
    auto const dims = inner.op.inputDimensions();
    Input x(dims), z(dims), zold(dims), u(dims), xpu(dims);
    x.setZero();
    z.setZero();
    u.setZero();
    xpu.setZero();
    float const absThresh = abstol * std::sqrt(float(Product(dims)));
    for (Index ii = 0; ii < iterLimit; ii++) {
      Input temp = x0 + x0.constant(ρ) * (z - u);
      x = inner.run(temp, x);
      xpu.device(dev) = x + u;
      zold = z;
      z = (*reg)(λ / ρ, xpu);
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
