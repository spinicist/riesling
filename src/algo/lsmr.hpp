#pragma once

#include "log.h"
#include "tensorOps.h"
#include "threads.h"
#include "types.h"

auto SymOrtho(float const a, float const b)
{
  if (b == 0.f) {
    return std::make_tuple(std::copysign(1.f, a), 0.f, std::abs(a));
  } else if (a == 0.f) {
    return std::make_tuple(0.f, std::copysign(1.f, b), std::abs(b));
  } else if (std::abs(b) > std::abs(a)) {
    auto const τ = a / b;
    float s = std::copysign(1.f, b) / std::sqrt(1.f + τ * τ);
    return std::make_tuple(s, s * τ, b / s);
  } else {
    auto const τ = b / a;
    float c = std::copysign(1.f, a) / std::sqrt(1.f + τ * τ);
    return std::make_tuple(c, c * τ, a / c);
  }
}

enum struct Reason
{
  XisZero = 0,
  AtolBtol,
  Atol,
  Cond1,
  Eps1,
  Eps2,
  Cond2,
  Iteratons
};

/* Based on https://github.com/PythonOptimizers/pykrylov/blob/master/pykrylov/lls/lsmr.py
 */
template <typename Op, typename Precond>
typename Op::Input lsmr(
  Index const &max_its,
  Op &op,
  typename Op::Output const &b,
  Precond const *M = nullptr, // Left preconditioner
  float const atol = 1.e-6f,
  float const btol = 1.e-6f,
  float const ctol = 1.e-6f,
  float const damp = 0.f)
{
  float const scale = Norm(b);
  Log::Print(
    FMT_STRING("Starting LSMR, scale {} Atol {} btol {} ctol {}"), scale, atol, btol, ctol);
  auto dev = Threads::GlobalDevice();
  // Allocate all memory
  using TI = typename Op::Input;
  using TO = typename Op::Output;
  auto const inDims = op.inputDimensions();
  auto const outDims = op.outputDimensions();

  TO Mu(outDims);
  TO u(outDims);
  Mu.device(dev) = b / b.constant(scale);
  u.device(dev) = M ? M->apply(Mu) : Mu;
  float β = sqrt(std::real(Dot(u, Mu)));
  Mu.device(dev) = Mu / Mu.constant(β);
  u.device(dev) = u / u.constant(β);

  TI v(inDims);
  v.device(dev) = op.Adj(u);
  float α = Norm(v);
  v.device(dev) = v / v.constant(α);

  TI h(inDims), h_(inDims), x(inDims);
  h.device(dev) = v;
  h_.setZero();
  x.setZero();

  // Initialize transformation variables. There are a lot
  float ζ_ = α * β;
  float α_ = α;
  float ρ = 1;
  float ρ_ = 1;
  float c_ = 1;
  float s_ = 0;

  // Initialize variables for ||r||
  float βdd = β;
  float βd = 0;
  float ρdold = 1;
  float τtildeold = 0;
  float θtilde = 0;
  float ζ = 0;
  float d = 0;

  // Initialize variables for estimation of ||A|| and cond(A)
  float normA2 = α * α;
  float maxrbar = 0;
  float minrbar = std::numeric_limits<float>::max();
  float const normb = β;
  Log::Print(FMT_STRING("Initial residual {}"), normb);

  for (Index ii = 0; ii < max_its; ii++) {
    // Bidiagonalization step
    Mu.device(dev) = op.A(v) - α * Mu;
    u.device(dev) = M ? M->apply(Mu) : Mu;
    β = sqrt(std::real(Dot(Mu, u)));
    Mu.device(dev) = Mu / Mu.constant(β);
    u.device(dev) = u / u.constant(β);

    v.device(dev) = op.Adj(u) - β * v;
    α = Norm(v);
    v.device(dev) = v / v.constant(α);

    float ch, sh, αh;
    std::tie(ch, sh, αh) = SymOrtho(α_, damp);

    // Construct rotation
    float ρold = ρ;
    float c, s;
    std::tie(c, s, ρ) = SymOrtho(αh, β);
    float θnew = s * α;
    α_ = c * α;

    // Use a plane rotation (Qbar_i) to turn R_i^T to R_i^bar

    float ρ_old = ρ_;
    float ζold = ζ;
    float θ_ = s_ * ρ;
    float ρtemp = c_ * ρ;
    std::tie(c_, s_, ρ_) = SymOrtho(c_ * ρ, θnew);
    ζ = c_ * ζ_;
    ζ_ = -s_ * ζ_;

    // Update h, h_h, x.

    h_.device(dev) = h - (θ_ * ρ / (ρold * ρ_old)) * h_;
    x.device(dev) = x + (ζ / (ρ * ρ_)) * h_;
    h.device(dev) = v - (θnew / ρ) * h;

    Log::Image(v, fmt::format(FMT_STRING("lsmr-v-{:02d}.nii"), ii));
    Log::Image(x, fmt::format(FMT_STRING("lsmr-x-{:02d}.nii"), ii));
    Log::Image(h, fmt::format(FMT_STRING("lsmr-h-{:02d}.nii"), ii));

    // Estimate of ||r||.

    // Apply rotation Qh_{k,2k+1}.
    float βacute = ch * βdd;
    float βcheck = -sh * βdd;

    // Apply rotation Q_{k,k+1}.
    float βh = c * βacute;
    βdd = -s * βacute;

    // Apply rotation Qtilde_{k-1}.
    // βd = βd_{k-1} here.

    float const θtildeold = θtilde;
    auto [ctildeold, stildeold, ρtildeold] = SymOrtho(ρdold, θ_);
    θtilde = stildeold * ρ_;
    ρdold = ctildeold * ρ_;
    βd = -stildeold * βd + ctildeold * βh;

    // βd   = βd_k here.
    // ρdold = ρd_k  here.

    τtildeold = (ζold - θtildeold * τtildeold) / ρtildeold;
    float const τd = (ζ - θtilde * τtildeold) / ρdold;
    d = d + βcheck * βcheck;
    float const normr = sqrt(d + pow(βd - τd, 2.f) + βdd * βdd);

    // Estimate ||A||.
    normA2 += β * β;
    float const normA = sqrt(normA2);
    normA2 += α * α;

    // Estimate cond(A).
    maxrbar = std::max(maxrbar, ρ_old);
    if (ii > 1) {
      minrbar = std::min(minrbar, ρ_old);
    }
    float const condA = std::max(maxrbar, ρtemp) / std::min(minrbar, ρtemp);

    Log::Print(FMT_STRING("LSMR {}: Residual {} Estimate cond(A) {}"), ii, normr, condA);

    // Convergence tests - go in pairs which check large/small values then the user tolerance
    float const normar = abs(ζ_);
    float const normx = Norm(x);

    if (1.f + (1.f / condA) <= 1.f) {
      Log::Print(FMT_STRING("Cond(A_) is very large"));
      break;
    }
    if ((1.f / condA) <= ctol) {
      Log::Print(FMT_STRING("Cond(A_) has exceeded limit"));
      break;
    }

    if (1.f + (normar / (normA * normr)) <= 1.f) {
      Log::Print(FMT_STRING("Least-squares solution reached machine precision"));
      break;
    }
    if ((normar / (normA * normr)) <= atol) {
      Log::Print(FMT_STRING("Least-squares < atol"));
      break;
    }

    if (normr <= (btol * normb + atol * normA * normx)) {
      Log::Print(FMT_STRING("Ax - b <= atol, btol"));
      break;
    }
    if ((1.f + normr / (normb + normA * normx)) <= 1.f) {
      Log::Print(FMT_STRING("Ax - b reached machine precision"));
      break;
    }
  }
  return x * x.constant(scale);
}
