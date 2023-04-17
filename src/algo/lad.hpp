#pragma once

#include "func/functor.hpp"
#include "log.hpp"
#include "op/operator.hpp"
#include "prox/thresh.hpp"
#include "signals.hpp"
#include "tensorOps.hpp"
#include "threads.hpp"

namespace rl {

template <typename Inner>
struct LAD
{
  using Input = typename Inner::Input;
  using Output = typename Inner::Output;

  Inner &inner;
  Index iterLimit = 8;
  float α = 1.f;  // Over-relaxation
  float μ = 10.f; // Primal-dual mismatch limit
  float τ = 2.f;  // Primal-dual mismatch rescale
  float abstol = 1.e-4f;
  float reltol = 1.e-4f;
  SoftThreshold<Input> S{1.f};

  Input run(Eigen::TensorMap<Output const> b, float ρ) const
  {
    auto dev = Threads::GlobalDevice();
    // Allocate all memory
    Input x(inner.op->ishape);
    auto const odims = inner.op->oshape;
    Output u(odims), z(odims), zold(odims), Ax_sub_b(odims);

    // Set initial values
    x.setZero();
    z.setZero();
    u.setZero();

    float const sqrtM = sqrt(Product(x.dimensions()));
    float const sqrtN = sqrt(Product(u.dimensions()));
    Log::Print(FMT_STRING("LAD ρ {} Abs Tol {} Rel Tol {}"), ρ, abstol, reltol);
    PushInterrupt();
    for (Index ii = 0; ii < iterLimit; ii++) {
      if (ii == 1) {
        inner.debug = true;
      } else {
        inner.debug = false;
      }
      x = inner.run(Output(b + z - u));
      Ax_sub_b = inner.op->cforward(x) - b;
      zold = z;
      z = S(1.f / ρ, Output(Ax_sub_b + u));
      u.device(dev) = u + Ax_sub_b - z;

      float const pNorm = Norm(Ax_sub_b - z);
      float const dNorm = ρ * Norm(z - zold);

      float const normx = Norm(x);
      float const normAx_sub_b = Norm(Ax_sub_b);
      float const normz = Norm(z);
      float const normu = Norm(u);

      float const pEps = abstol * sqrtM + reltol * std::max(normx, normz);
      float const dEps = abstol * sqrtN + reltol * ρ * normu;

      Log::Tensor(x, fmt::format("admm-x-{:02d}", ii));
      Log::Print(
        FMT_STRING("ADMM {:02d}: Primal || {} ε {} Dual || {} ε {} |x| {} |Ax - b| {} |z| {} |u| {}"),
        ii,
        pNorm,
        pEps,
        dNorm,
        dEps,
        normx,
        normAx_sub_b,
        normz,
        normu);
      if ((pNorm < pEps) && (dNorm < dEps)) {
        Log::Print("Primal and dual convergence achieved, stopping");
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
