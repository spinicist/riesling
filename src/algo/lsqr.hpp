#pragma once

#include "common.hpp"
#include "func/functor.hpp"
#include "log.hpp"
#include "tensorOps.hpp"
#include "threads.hpp"
#include "types.hpp"
#include "bidiag.hpp"

namespace rl {
/* Based on https://github.com/PythonOptimizers/pykrylov/blob/master/pykrylov/lls/lsqr.py
 */

/*
 * LSQR with arbitrary regularization, i.e. Solve (A'A + λI)x = A'b + c with warm start
 */
template <typename Op>
struct LSQR
{
  using Input = typename Op::Input;
  using Output = typename Op::Output;

  std::shared_ptr<Op> op;
  std::shared_ptr<Functor<Output>> M = std::make_shared<IdentityFunctor<Output>>(); // Left pre-conditioner
  Index iterLimit;
  float aTol = 1.e-6f;
  float bTol = 1.e-6f;
  float cTol = 1.e-6f;
  bool const debug = false;

  Input run(Eigen::TensorMap<Output const> b, float const λ = 0.f, Input const &x0 = Input(), Input const &cc = Input()) const
  {
    auto dev = Threads::GlobalDevice();
    // Allocate all memory
    auto const inDims = op->inputDimensions();
    auto const outDims = op->outputDimensions();

    // Workspace variables
    Output Mu(outDims), u(outDims);
    Input x(inDims), v(inDims), w(inDims), ur;
    float α = 0.f, β = 0.f;
    BidiagInit(op, M, Mu, u, ur, v, α, β, λ, x, b, x0, cc, dev);
    w.device(dev) = v;

    float ρ̅ = α;
    float ɸ̅ = β;
    float normb = β, xxnorm = 0.f, ddnorm = 0.f, res2 = 0.f, z = 0.f;
    float normA = 0.f;
    float cs2 = -1.f;
    float sn2 = 0.f;

    if (debug) {
      Log::Tensor(x, "lsqr-x-init");
      Log::Tensor(v, "lsqr-v-init");
    }

    Log::Print(FMT_STRING("LSQR    α {:5.3E} β {:5.3E} λ {}{}"), α, β, λ, x0.size() ? " with initial guess" : "");

    for (Index ii = 0; ii < iterLimit; ii++) {
      Bidiag(op, M, Mu, u, ur, v, α, β, λ, dev);

      float c, s, ρ;
      float ψ = 0.f;
      if (λ == 0.f || ur.size()) {
        std::tie(c, s, ρ) = StableGivens(ρ̅, β);
      } else {
        // Deal with regularization
        auto [c1, s1, ρ̅1] = StableGivens(ρ̅, λ);
        ψ = s1 * ɸ̅;
        ɸ̅ = c1 * ɸ̅;
        std::tie(c, s, ρ) = StableGivens(ρ̅1, β);
      }
      float const ɸ = c * ɸ̅;
      ɸ̅ = s * ɸ̅;
      float const τ = s * ɸ;
      float const θ = s * α;
      ρ̅ = -c * α;
      x.device(dev) = x + w * w.constant(ɸ / ρ);
      w.device(dev) = v - w * w.constant(θ / ρ);

      if (debug) {
        Log::Tensor(x, fmt::format(FMT_STRING("lsqr-x-{:02d}"), ii));
        Log::Tensor(v, fmt::format(FMT_STRING("lsqr-w-{:02d}"), ii));
      }

      // Estimate norms
      float const δ = sn2 * ρ;
      float const ɣ̅ = -cs2 * ρ;
      float const rhs = ɸ - δ * z;
      float const zbar = rhs / ɣ̅;
      float const normx = std::sqrt(xxnorm + zbar * zbar);
      float ɣ;
      std::tie(cs2, sn2, ɣ) = StableGivens(ɣ̅, θ);
      z = rhs / ɣ;
      xxnorm += z * z;
      ddnorm = ddnorm + Norm2(w) / (ρ * ρ);

      normA = std::sqrt(normA * normA + α * α + β * β + λ * λ);
      float const condA = normA * std::sqrt(ddnorm);
      float const res1 = ɸ̅ * ɸ̅;
      res2 = res2 + ψ * ψ;
      float const normr = std::sqrt(res1 + res2);
      float const normAr = α * std::abs(τ);

      Log::Print(
        FMT_STRING("LSQR {:02d} α {:5.3E} β {:5.3E} |r| {:5.3E} |A'r| {:5.3E} |A| {:5.3E} cond(A) "
                   "{:5.3E} |x| {:5.3E}"),
        ii,
        α,
        β,
        normr,
        normAr,
        normA,
        condA,
        normx);

      if (1.f + (1.f / condA) <= 1.f) {
        Log::Print(FMT_STRING("Cond(A) is very large"));
        break;
      }
      if ((1.f / condA) <= cTol) {
        Log::Print(FMT_STRING("Cond(A) has exceeded limit"));
        break;
      }

      if (1.f + (normAr / (normA * normr)) <= 1.f) {
        Log::Print(FMT_STRING("Least-squares solution reached machine precision"));
        break;
      }
      if ((normAr / (normA * normr)) <= aTol) {
        Log::Print(FMT_STRING("Least-squares = {:5.3E} < {:5.3E}"), normAr / (normA * normr), aTol);
        break;
      }

      if (normr <= (bTol * normb + aTol * normA * normx)) {
        Log::Print(FMT_STRING("Ax - b <= aTol, bTol"));
        break;
      }
      if ((1.f + normr / (normb + normA * normx)) <= 1.f) {
        Log::Print(FMT_STRING("Ax - b reached machine precision"));
        break;
      }
    }

    return x;
  }
};

} // namespace rl
