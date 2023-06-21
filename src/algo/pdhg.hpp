#pragma once

#include "prox/prox.hpp"
#include "log.hpp"
#include "tensorOps.hpp"
#include "threads.hpp"
#include "common.hpp"

namespace rl {

template <typename Op>
struct PrimalDualHybridGradient
{
  using Input = typename Ops::Input;
  using Output = typename Ops::Output;
  using Scalar = typename Ops::Scalar;

  std::shared_ptr<Op> op;
  Cx4 P;
  std::shared_ptr<Prox<Input>> prox;
  Index iterLimit = 8;

  Input run(Eigen::TensorMap<Output const> b, float τ = 1.f /* Primal step size */) const
  {
    auto dev = Threads::GlobalDevice();

    auto const inDims = op->ishape;
    auto const outDims = op->oshape;
    CheckDimsEqual(b.dimensions(), outDims);
    Output u(outDims);
    u.setZero();
    Input x(inDims), xbar(inDims), xold(inDims);
    xbar.setZero();
    x.setZero();

    float σ = 1.f;                       // Dual Step-size
    float const pmin = Minimum(P.abs()); // For updating θ
    Log::Print("PDHG pmin {}", pmin);
    for (Index ii = 0; ii < iterLimit; ii++) {
      xold.device(dev) = x;
      u.device(dev) = (u + u.constant(σ) * P * (op->forward(xbar) - b)) / (u.constant(1.f) + u.constant(σ) * P);
      x.device(dev) = x - x.constant(τ) * op->adjoint(u);
      Log::Tensor(x, fmt::format("pdhg-xa-{:02d}", ii));
      x.device(dev) = (*prox)(τ, x);
      Log::Tensor(x, fmt::format("pdhg-x-{:02d}", ii));
      float const θ = 1.f / (1.f + 2.f * σ * pmin);
      xold.device(dev) = x - xold;
      float const normr = Norm(xold / xold.constant(std::sqrt(τ)));
      xbar.device(dev) = x + θ * (xold);
      Log::Print("PDHG {:02d}: |r| {} σ {} τ {} θ {}", ii, normr, σ, τ, θ);
      σ *= θ;
      τ /= θ;
    }
    return x;
  }
};

} // namespace rl
