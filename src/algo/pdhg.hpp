#pragma once

#include "func/functor.hpp"
#include "log.hpp"
#include "tensorOps.hpp"
#include "threads.hpp"
#include "common.hpp"

namespace rl {

template <typename Op>
struct PrimalDualHybridGradient
{
  using Input = typename Op::Input;
  using Output = typename Op::Output;
  using Scalar = typename Op::Scalar;

  std::shared_ptr<Op> op;
  Cx4 P;
  std::shared_ptr<Prox<Input>> prox;
  Index iterLimit = 8;

  Input run(Eigen::TensorMap<Output const> b, float τ = 1.f /* Primal step size */) const
  {
    auto dev = Threads::GlobalDevice();

    auto const inDims = op->inputDimensions();
    auto const outDims = op->outputDimensions();
    CheckDimsEqual(b.dimensions(), outDims);
    Output u(outDims), utemp(outDims);
    u.setZero();
    Input x(inDims), xbar(inDims), xold(inDims);
    xbar.setZero();
    x.setZero();

    float σ = 1.f;                       // Dual Step-size
    float const pmin = Minimum(P.abs()); // For updating θ
    Log::Print(FMT_STRING("PDHG pmin {}"), pmin);
    for (Index ii = 0; ii < iterLimit; ii++) {
      xold.device(dev) = x;
      u.device(dev) = (u + u.constant(σ) * P * (op->forward(x) - b)) / (u.constant(1.f) + u.constant(σ) * P);
      x.device(dev) = x - x.constant(τ) * op->adjoint(u);
      x.device(dev) = (*prox)(τ, x);
      float const θ = 1.f / std::sqrt(1.f + 2.f * pmin);
      xold.device(dev) = x - xold;
      float const normr = Norm(xold / xold.constant(std::sqrt(τ)));
      // xbar.device(dev) = x + θ * (xold);

      Log::Tensor(x, fmt::format("pdhg-x-{:02d}", ii));
      Log::Print(FMT_STRING("PDHG {:02d}: |r| {} σ {} τ {} θ {}"), ii, normr, σ, τ, θ);
      σ *= θ;
      τ /= θ;
    }
    return x;
  }
};

} // namespace rl
