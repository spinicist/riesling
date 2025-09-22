#include "lsmr.hpp"

#include "../log/log.hpp"
#include "bidiag.hpp"
#include "common.hpp"
#include "iter.hpp"

namespace rl {

auto LSMR::run(Vector const &b, Vector const &x0) const -> Vector
{
  return run(CMap{b.data(), b.rows()}, CMap{x0.data(), x0.rows()});
}

/* Based on https://github.com/PythonOptimizers/pykrylov/blob/master/pykrylov/lls/lsmr.py
 */
auto LSMR::run(CMap b, CMap x0) const -> Vector
{
  Log::Print("LSMR", "λ {}", opts.λ);
  if (opts.imax < 1) { throw Log::Failure("LSMR", "Requires at least 1 iteration"); }
  Index const rows = A->rows();
  Index const cols = A->cols();
  if (rows < 1 || cols < 1) { throw Log::Failure("LSMR", "Invalid operator size rows {} cols {}", rows, cols); }
  if (b.rows() != rows) { throw Log::Failure("LSMR", "b had size {} expected {}", b.rows(), rows); }
  Vector h(cols), h̅(cols), x(cols);
  Bidiag bd(A, Minv, Ninv, x, b, x0);
  h = bd.v;
  h̅.setZero();

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
  float       minρ̅ = std::numeric_limits<float>::max();
  float const normb = bd.β;

  Log::Print("LSMR", "IT |x|       |r|       Tol       |A'r|     Tol       |A|       cond(A)");
  Log::Print("LSMR", "{:02d} {:4.3E} {:4.3E} {:4.3E} {:4.3E} {:4.3E}", 0, ParallelNorm(x), normb, 0.f, std::fabs(ζ̅), 0.f);
  Iterating::Starting();
  for (Index ii = 0; ii < opts.imax; ii++) {
    bd.next();

    float const ρold = ρ;
    float       c, s, ĉ = 1.f, ŝ = 0.f;
    if (opts.λ == 0.f) {
      std::tie(c, s, ρ) = StableGivens(α̅, bd.β);
    } else {
      float α̂;
      std::tie(ĉ, ŝ, α̂) = StableGivens(α̅, opts.λ);
      std::tie(c, s, ρ) = StableGivens(α̂, bd.β);
    }
    float θnew = s * bd.α;
    α̅ = c * bd.α;

    // Use a plane rotation (Qbar_i) to turn R_i^T to R_i^bar
    float ρ̅old = ρ̅;
    float ζold = ζ;
    float θ̅ = s̅ * ρ;
    float ρtemp = c̅ * ρ;
    std::tie(c̅, s̅, ρ̅) = StableGivens(ρtemp, θnew);
    ζ = c̅ * ζ̅;
    ζ̅ = -s̅ * ζ̅;

    // Update h, h̅, x.
    h̅.device(Threads::CoreDevice()) = h - (θ̅ * ρ / (ρold * ρ̅old)) * h̅;
    x.device(Threads::CoreDevice()) = x + (ζ / (ρ * ρ̅)) * h̅;
    h.device(Threads::CoreDevice()) = bd.v - (θnew / ρ) * h;

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
    normA2 += bd.β * bd.β;
    float const normA = std::sqrt(normA2);
    normA2 += bd.α * bd.α;

    // Estimate cond(A).
    maxρ̅ = std::max(maxρ̅, ρ̅old);
    if (ii > 1) { minρ̅ = std::min(minρ̅, ρ̅old); }
    float const condA = std::max(maxρ̅, ρtemp) / std::min(minρ̅, ρtemp);

    // Convergence tests - go in pairs which check large/small values then the user tolerance
    float const normAr = abs(ζ̅);
    float const normx = ParallelNorm(x);
    float const thresh1 = opts.bTol * normb + opts.aTol * normA * normx;
    float const thresh2 = opts.aTol * (normA * normr);
    Log::Print("LSMR", "{:02d} {:4.3E} {:4.3E} {:4.3E} {:4.3E} {:4.3E} {:4.3E} {:4.3E}", ii + 1, normx, normr, thresh1, normAr, thresh2, normA, condA);
    if (debug) { debug(ii, x, bd.v); }
    if (normr <= thresh1) {
      Log::Print("LSMR", "Ax - b <= aTol, bTol");
      break;
    }
    if (normAr <= thresh2) {
      Log::Print("LSMR", "Least-squares = {:4.3E} < aTol = {:4.3E}", normAr, thresh2);
      break;
    }
    if (1.f + (1.f / condA) <= 1.f) {
      Log::Print("LSMR", "Cond(A) is very large");
      break;
    }
    if ((1.f / condA) <= opts.cTol) {
      Log::Print("LSMR", "Cond(A) has exceeded limit");
      break;
    }
    if (1.f + (normAr / (normA * normr)) <= 1.f) {
      Log::Print("LSMR", "Least-squares solution reached machine precision");
      break;
    }
    if ((1.f + normr / (normb + normA * normx)) <= 1.f) {
      Log::Print("LSMR", "Ax - b reached machine precision");
      break;
    }
    if (Iterating::ShouldStop("LSMR")) { break; }
  }
  Iterating::Finished();
  return x;
}

} // namespace rl
