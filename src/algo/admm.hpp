#pragma once

#include "func/functor.hpp"
#include "op/operator.hpp"
#include "log.hpp"
#include "signals.hpp"
#include "tensorOps.hpp"
#include "threads.hpp"

namespace rl {

template<typename Op>
struct Regularizer {
  std::shared_ptr<Prox<typename Op::Output>> prox;
  std::shared_ptr<Op> op;
};

template <typename Inner, typename RegOp>
struct ADMM
{
  using Input = typename Inner::Input;
  using Output = typename Inner::Output;

  Inner &inner;
  Regularizer<RegOp> reg;
  Index iterLimit = 8;
  float α = 1.f;  // Over-relaxation
  float μ = 10.f; // Primal-dual mismatch limit
  float τ = 2.f;  // Primal-dual mismatch rescale
  float abstol = 1.e-3f;
  float reltol = 1.e-3f;

  Input run(Eigen::TensorMap<Output const> b, float ρ) const
  {
    auto dev = Threads::GlobalDevice();
    // Allocate all memory
    Input x(inner.op->inputDimensions());
    auto const dims = reg.op->outputDimensions();
    typename RegOp::Output u(dims), z(dims), zold(dims), Fx(dims), Fxpu(dims);

    // Set initial values
    x.setZero();
    z.setZero();
    u.setZero();

    float const sqrtM = sqrt(Product(x.dimensions()));
    float const sqrtN = sqrt(Product(u.dimensions()));
    Log::Print(FMT_STRING("ADMM ρ {}"), ρ);
    PushInterrupt();
    for (Index ii = 0; ii < iterLimit; ii++) {
      if (ii == 1) {
        inner.debug = true;
      } else {
        inner.debug = false;
      }
      x = inner.run(b, ρ, x, (z - u));
      Fx = reg.op->forward(x);
      Fxpu.device(dev) = Fx * Fx.constant(α) + z * z.constant(1.f - α) + u;
      zold = z;
      z = (*reg.prox)(1.f / ρ, Fxpu);
      u.device(dev) = Fxpu - z;

      float const pNorm = Norm(Fx - z);
      float const dNorm = ρ * Norm(z - zold);

      float const normx = Norm(x);
      float const normFx = Norm(Fx);
      float const normz = Norm(z);
      float const normu = Norm(u);

      float const pEps = abstol * sqrtM + reltol * std::max(normx, normz);
      float const dEps = abstol * sqrtN + reltol * ρ * normu;

      Log::Tensor(x, fmt::format("admm-x-{:02d}", ii));
      Log::Tensor(z, fmt::format("admm-z-{:02d}", ii));
      Log::Tensor(Fx, fmt::format("admm-Fx-{:02d}", ii));
      Log::Tensor(Fxpu, fmt::format("admm-Fxpu-{:02d}", ii));
      Log::Tensor(u, fmt::format("admm-u-{:02d}", ii));
      Log::Print(
        FMT_STRING("ADMM {:02d}: Primal || {} ε {} Dual || {} ε {} |x| {} |Fx| {} |z| {} |u| {}"),
        ii,
        pNorm,
        pEps,
        dNorm,
        dEps,
        normx,
        normFx,
        normz,
        normu);
      if ((pNorm < pEps) && (dNorm < dEps)) {
        break;
      }
      if (pNorm > μ * dNorm) {
        ρ *= τ;
        u /= u.constant(τ);
        Log::Print(FMT_STRING("Primal norm outside limit {}, rescaled ρ to {} |u| {}"), μ * dNorm, ρ, Norm(u));
      } else if (dNorm > μ * pNorm) {
        ρ /= τ;
        u *= u.constant(τ);
        Log::Print(FMT_STRING("Dual norm outside limit {}, rescaled ρ to {} |u| {}"), μ * pNorm, ρ, Norm(u));
      }
      if (InterruptReceived()) {
        break;
      }
    }
    PopInterrupt();
    return x;
  }
};

} // namespace rl
