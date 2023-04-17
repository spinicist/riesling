#pragma once

#include "func/functor.hpp"
#include "log.hpp"
#include "op/operator.hpp"
#include "signals.hpp"
#include "tensorOps.hpp"
#include "threads.hpp"

namespace rl {

auto ADMM::run(Cx *bdata, float ρ) const -> Vector;
{
  Map const b(bdata, lsq->cols);

  std::shared_ptr<Op> reg = std::make_shared<VStack>(reg_ops);
  std::shared_ptr<Op> ρscale = std::make_shared<Scale>(reg->rows(), ρ);
  std::shared_ptr<Op> ρReg = std::make_shared<Concatenate>(reg, ρscale);
  std::shared_ptr<Op> A = std::make_shared<VStack>({lsq, ρReg});

    Vector x(lsq->rows()), u()

   u(dims), z(dims), zold(dims), Fx(dims), Fxpu(dims);

  // Set initial values
  x.setZero();
  z.setZero();
  u.setZero();

  float const sqrtM = std::sqrt(Product(x.dimensions()));
  float const sqrtN = std::sqrt(Product(u.dimensions()));
  Log::Print(FMT_STRING("ADMM ρ {} Abs Tol {} Rel Tol {}"), ρ, abstol, reltol);
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
    (*reg.prox)(1.f / ρ, Fxpu, z);
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
