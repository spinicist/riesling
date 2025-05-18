#pragma once

#include "bidiag.cuh"
#include "../op/op.cuh"
#include "rl/algo/givens.hpp"
#include "rl/algo/iter.hpp"
#include "rl/log/log.hpp"

namespace gw {

template <typename T, int xRank, int yRank> struct LSMR
{
  struct Opts
  {
    int   imax = 4;
    float aTol = 1.e-6f;
    float bTol = 1.e-6f;
    float cTol = 1.e-6f;
    float λ = 0.f;
  };

  Op<T, xRank, yRank> const *A;
  Op<T, yRank, yRank> const *Minv = nullptr; // Left Pre-conditioner
  Op<T, xRank, xRank> const *Ninv = nullptr; // Right Pre-conditioner

  Opts opts;

  void run(DTensor<T, yRank> const &b, DTensor<T, xRank> &x) const
  {
    rl::Log::Print("LSMR", "λ {}", opts.λ);
    if (opts.imax < 1) { throw rl::Log::Failure("LSMR", "Requires at least 1 iteration"); }
    DTensor<T, xRank>       h(x.span), h̅(x.span);
    Bidiag<T, xRank, yRank> bd(A, Minv, Ninv, x, b);
    thrust::copy(bd.v.vec.begin(), bd.v.vec.end(), h.vec.begin());
    thrust::fill(h̅.vec.begin(), h̅.vec.end(), T(0));

    // Initialize transformation variables. There are a lot
    float ζ̅ = bd.α * bd.β;
    float α̅ = bd.α;
    float ρ = 1;
    float ρ̅ = 1;
    float c̅ = 1;
    float s̅ = 0;

    // Initialize variables for ||r||
    float β̈ = bd.β;
    float β̇ = 0;
    float ρ̇old = 1;
    float τ̃old = 0;
    float θ̃ = 0;
    float ζ = 0;
    float d = 0;

    // Initialize variables for estimation of ||A|| and cond(A)
    float       normA2 = bd.α * bd.α;
    float       maxρ̅ = 0;
    float       minρ̅ = cuda::std::numeric_limits<float>::max();
    float const normb = bd.β;

    rl::Log::Print("LSMR", "IT |x|       |r|       Tol       |A'r|     Tol       |A|       cond(A)");
    rl::Log::Print("LSMR", "{:02d} {:4.3E} {:4.3E} {:4.3E} {:4.3E} {:4.3E}", 0, FLOAT_FROM(gw::CuNorm(x.vec)), normb, 0.f,
                   cuda::std::fabs(ζ̅), 0.f);
    rl::Iterating::Starting();
    for (int ii = 0; ii < opts.imax; ii++) {
      bd.next();

      float const ρold = ρ;
      float       c, s, ĉ = 1.f, ŝ = 0.f;
      if (opts.λ == 0.f) {
        std::tie(c, s, ρ) = rl::StableGivens(α̅, bd.β);
      } else {
        float α̂;
        std::tie(ĉ, ŝ, α̂) = rl::StableGivens(α̅, opts.λ);
        std::tie(c, s, ρ) = rl::StableGivens(α̂, bd.β);
      }
      float θnew = s * bd.α;
      α̅ = c * bd.α;

      // Use a plane rotation (Qbar_i) to turn R_i^T to R_i^bar
      float ρ̅old = ρ̅;
      float ζold = ζ;
      float θ̅ = s̅ * ρ;
      float ρtemp = c̅ * ρ;
      auto const [c̅, s̅, ρ̅] = rl::StableGivens(ρtemp, θnew);
      ζ = c̅ * ζ̅;
      ζ̅ = -s̅ * ζ̅;

      // Update h, h̅, x.
      thrust::transform(
        h.vec.begin(), h.vec.end(), h̅.vec.begin(), h̅.vec.begin(),
        [z = FLOAT_TO(θ̅ * ρ / (ρold * ρ̅old))] __device__(CuCx<TDev> const h, CuCx<TDev> const h̅) { return h - z * h̅; });
      thrust::transform(x.vec.begin(), x.vec.end(), h̅.vec.begin(), x.vec.begin(),
                        [z = FLOAT_TO(ζ / (ρ * ρ̅))] __device__(CuCx<TDev> const x, CuCx<TDev> const h̅) { return x + z * h̅; });
      thrust::transform(bd.v.vec.begin(), bd.v.vec.end(), h.vec.begin(), h̅.vec.begin(),
                        [z = FLOAT_TO(θnew / ρ)] __device__(CuCx<TDev> const v, CuCx<TDev> h) { return v - z * h; });

      // Estimate of |r|.
      float const β́ = ĉ * β̈;
      float const β̆ = -ŝ * β̈;

      float const β̂ = c * β́;
      β̈ = -s * β́;

      float const θ̃old = θ̃;
      auto [c̃old, s̃old, ρ̃old] = rl::StableGivens(ρ̇old, θ̅);
      θ̃ = s̃old * ρ̅;
      ρ̇old = c̃old * ρ̅;
      β̇ = -s̃old * β̇ + c̃old * β̂;

      τ̃old = (ζold - θ̃old * τ̃old) / ρ̃old;
      float const τ̇ = (ζ - θ̃ * τ̃old) / ρ̇old;
      d = d + β̆ * β̆;
      float const normr = cuda::std::sqrt(d + cuda::std::pow(β̇ - τ̇, 2) + β̈ * β̈);
      // Estimate ||A||.
      normA2 += bd.β * bd.β;
      float const normA = cuda::std::sqrt(normA2);
      normA2 += bd.α * bd.α;

      // Estimate cond(A).
      maxρ̅ = cuda::std::max(maxρ̅, ρ̅old);
      if (ii > 1) { minρ̅ = cuda::std::min(minρ̅, ρ̅old); }
      float const condA = cuda::std::max(maxρ̅, ρtemp) / cuda::std::min(minρ̅, ρtemp);

      // Convergence tests - go in pairs which check large/small values then the user tolerance
      float const normAr = cuda::std::abs(ζ̅);
      float const normx = FLOAT_FROM(gw::CuNorm(x.vec));
      float const thresh1 = opts.bTol * normb + opts.aTol * normA * normx;
      float const thresh2 = opts.aTol * (normA * normr);
      rl::Log::Print("LSMR", "{:02d} {:4.3E} {:4.3E} {:4.3E} {:4.3E} {:4.3E} {:4.3E} {:4.3E}", ii + 1, normx, normr, thresh1,
                     normAr, thresh2, normA, condA);
      if (normr <= thresh1) {
        rl::Log::Print("LSMR", "Ax - b <= aTol, bTol");
        break;
      }
      if (normAr <= thresh2) {
        rl::Log::Print("LSMR", "Least-squares = {:4.3E} < aTol = {:4.3E}", normAr, thresh2);
        break;
      }
      if (1.f + (1.f / condA) <= 1.f) {
        rl::Log::Print("LSMR", "Cond(A) is very large");
        break;
      }
      if ((1.f / condA) <= opts.cTol) {
        rl::Log::Print("LSMR", "Cond(A) has exceeded limit");
        break;
      }
      if (1.f + (normAr / (normA * normr)) <= 1.f) {
        rl::Log::Print("LSMR", "Least-squares solution reached machine precision");
        break;
      }
      if ((1.f + normr / (normb + normA * normx)) <= 1.f) {
        rl::Log::Print("LSMR", "Ax - b reached machine precision");
        break;
      }
      if (rl::Iterating::ShouldStop("LSMR")) { break; }
    }
    rl::Iterating::Finished();
  }
};

} // namespace gw
