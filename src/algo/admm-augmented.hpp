#pragma once

#include "cg.hpp"
#include "log.h"
#include "tensorOps.h"
#include "threads.h"

namespace rl {

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

template <typename Inner>
struct AugmentedADMM
{
  using Input = typename Inner::Input;

  Inner &inner;
  std::function<Input(Input const &)> const &reg;
  Index iterLimit;
  float rho = 0.1;
  float abstol = 1.e-3f;
  float reltol = 1.e-3f;

  Input run(Input const &x0) const
  {
    Log::Print(FMT_STRING("ADMM-CG rho {}"), rho);
    auto dev = Threads::GlobalDevice();
    // Allocate all memory
    auto const dims = inner.op.inputDimensions();
    Index const N = Product(dims);
    Input x(dims), z(dims), zold(dims), u(dims), xpu(dims);
    x.setZero();
    z.setZero();
    u.setZero();
    xpu.setZero();
    for (Index ii = 0; ii < iterLimit; ii++) {
      x = inner.run(x0 + x0.constant(rho) * (z - u), x);
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
        FMT_STRING("ADMM {:02d}: Primal || {} ε {} Dual || {} ε {}"),
        ii,
        norm_prim,
        eps_prim,
        norm_dual,
        eps_dual);
      if ((norm_prim < eps_prim) && (norm_dual < eps_dual)) {
        break;
      }
      float const mu = 10.f;
      if (norm_prim > mu * norm_dual) {
        Log::Print(FMT_STRING("Primal norm is outside limit {}, consider changing rho"), mu * norm_dual);
      } else if (norm_dual > mu * norm_prim) {
        Log::Print(FMT_STRING("Dual norm is outside limit {}, consider changing rho"), mu * norm_prim);
      }
    }
    return x;
  }
};

} // namespace rl
