#pragma once

#include "bidiag.hpp"
#include "common.hpp"
#include "func/functor.hpp"
#include "log.hpp"
#include "op/identity.hpp"
#include "signals.hpp"
#include "tensorOps.hpp"
#include "threads.hpp"
#include "types.hpp"

namespace rl {
/* Based on https://github.com/PythonOptimizers/pykrylov/blob/master/pykrylov/lls/lsmr.py
 */

/*
 * LSMR with arbitrary regularization, i.e. Solve (A'A + λI)x = A'b + c with warm start
 */
template <typename Op, typename Opλ = Operator<typename Op::Scalar, Op::InputRank, Op::InputRank>>
struct LSMR
{
  using Scalar = typename Op::Scalar;
  using Input = typename Op::Input;
  using Output = typename Op::Output;
  using Outputλ = typename Opλ::Output;
  using Pre = Operator<typename Op::Scalar, Op::OutputRank, Op::OutputRank>;

  std::shared_ptr<Op> op;
  std::shared_ptr<Pre> M = std::make_shared<Identity<Scalar, Op::OutputRank>>(op->outputDimensions()); // Left pre-conditioner
  Index iterLimit = 8;
  float aTol = 1.e-6f;
  float bTol = 1.e-6f;
  float cTol = 1.e-6f;
  bool debug = false;
  std::shared_ptr<Opλ> opλ = std::make_shared<Identity<Scalar, Op::InputRank>>(op->inputDimensions());

  Input run(Eigen::TensorMap<Output const> const &b, float const λ = 0.f, Input const &x0 = Input(), Outputλ const &b0 = Input()) const
  {
    auto dev = Threads::GlobalDevice();

    auto const inDims = op->inputDimensions();
    auto const outDims = op->outputDimensions();
    CheckDimsEqual(b.dimensions(), outDims);
    Output Mu(outDims), u(outDims);
    Input v(inDims), h(inDims), h̅(inDims), x(inDims);
    Outputλ uλ(b0.dimensions());
    float α = 0.f, β = 0.f;
    BidiagInit(op, M, Mu, u, v, α, β, λ, opλ, uλ, x, b, x0, b0, dev);
    h.device(dev) = v;
    h̅.setZero();

    // Initialize transformation variables. There are a lot
    float ζ̅ = α * β;
    float α̅ = α;
    float ρ = 1;
    float ρ̅ = 1;
    float c̅ = 1;
    float s̅ = 0;

    // Initialize variables for ||r||
    float β̈ = β;
    float β̇ = 0;
    float ρ̇old = 1;
    float τ̃old = 0;
    float θ̃ = 0;
    float ζ = 0;
    float d = 0;

    // Initialize variables for estimation of ||A|| and cond(A)
    float normA2 = α * α;
    float maxρ̅ = 0;
    float minρ̅ = std::numeric_limits<float>::max();
    float const normb = β;

    if (debug) {
      Log::Tensor(v, "lsmr-v-init");
      Log::Tensor(x, "lsmr-x-init");
    }

    Log::Print(FMT_STRING("LSMR α {:5.3E} β {:5.3E} λ {}"), α, β, λ);
    Log::Print("IT α         β         |r|       |A'r|     |A|       cond(A)   |x|");
    PushInterrupt();
    for (Index ii = 0; ii < iterLimit; ii++) {
      Bidiag(op, M, Mu, u, v, α, β, λ, opλ, uλ, dev);

      float const ρold = ρ;
      float c, s, ĉ = 1.f, ŝ = 0.f;
      if (λ == 0.f || uλ.size()) {
        std::tie(c, s, ρ) = StableGivens(α̅, β);
      } else {
        float α̂;
        std::tie(ĉ, ŝ, α̂) = StableGivens(α̅, λ);
        std::tie(c, s, ρ) = StableGivens(α̂, β);
      }
      float θnew = s * α;
      α̅ = c * α;

      // Use a plane rotation (Qbar_i) to turn R_i^T to R_i^bar
      float ρ̅old = ρ̅;
      float ζold = ζ;
      float θ̅ = s̅ * ρ;
      float ρtemp = c̅ * ρ;
      std::tie(c̅, s̅, ρ̅) = StableGivens(ρtemp, θnew);
      ζ = c̅ * ζ̅;
      ζ̅ = -s̅ * ζ̅;

      // Update h, h̅, x.
      h̅.device(dev) = h - (θ̅ * ρ / (ρold * ρ̅old)) * h̅;
      x.device(dev) = x + (ζ / (ρ * ρ̅)) * h̅;
      h.device(dev) = v - (θnew / ρ) * h;

      // Estimate of |r|.
      float const β́ = ĉ * β̈;
      float const β̆ = -ŝ * β̈;

      float const β̂ = c * β́;
      β̈ = -s * β́;

      float const θ̃old = θ̃;
      auto [c̃old, s̃old, ρ̃old] = StableGivens(ρ̇old, θ̅);
      θ̃ = s̃old * ρ̅;
      ρ̇old = c̃old * ρ̅;
      β̇ = -s̃old * β̇ + c̃old * β̂;

      τ̃old = (ζold - θ̃old * τ̃old) / ρ̃old;
      float const τ̇ = (ζ - θ̃ * τ̃old) / ρ̇old;
      d = d + β̆ * β̆;
      float const normr = std::sqrt(d + std::pow(β̇ - τ̇, 2) + β̈ * β̈);
      // Estimate ||A||.
      normA2 += β * β;
      float const normA = std::sqrt(normA2);
      normA2 += α * α;

      // Estimate cond(A).
      maxρ̅ = std::max(maxρ̅, ρ̅old);
      if (ii > 1) {
        minρ̅ = std::min(minρ̅, ρ̅old);
      }
      float const condA = std::max(maxρ̅, ρtemp) / std::min(minρ̅, ρtemp);

      // Convergence tests - go in pairs which check large/small values then the user tolerance
      float const normAr = abs(ζ̅);
      float const normx = Norm(x);

      Log::Print(
        FMT_STRING("{:02d} {:5.3E} {:5.3E} {:5.3E} {:5.3E} {:5.3E} {:5.3E} {:5.3E}"),
        ii,
        α,
        β,
        normr,
        normAr,
        normA,
        condA,
        normx);

      if (debug) {
        Log::Tensor(x, fmt::format(FMT_STRING("lsmr-x-{:02d}"), ii));
        Log::Tensor(v, fmt::format(FMT_STRING("lsmr-v-{:02d}"), ii));
        Log::Tensor(h̅, fmt::format(FMT_STRING("lsmr-hbar-{:02d}"), ii));
        // Log::Tensor(uλ, fmt::format(FMT_STRING("lsmr-ul-{:02d}"), ii));
      }

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
        Log::Print(FMT_STRING("Least-squares = {:5.3E} < aTol = {:5.3E}"), normAr / (normA * normr), aTol);
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
      if (InterruptReceived()) {
        break;
      }
    }
    PopInterrupt();
    return x;
  }
};

} // namespace rl
