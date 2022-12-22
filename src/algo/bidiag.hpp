#pragma once

#include "common.hpp"

namespace rl {

namespace {

inline auto StableGivens(float const a, float const b)
{
  float c, s, ρ;
  if (b == 0.f) {
    if (a == 0.f) {
      c = 1.f;
    } else {
      c = std::copysign(1.f, a);
    }
    s = 0.f;
    ρ = std::abs(a);
  } else if (a == 0.f) {
    c = 0.f;
    s = std::copysign(1.f, b);
    ρ = std::abs(b);
  } else if (std::abs(b) > std::abs(a)) {
    auto const τ = a / b;
    s = std::copysign(1.f / std::sqrt(1.f + τ * τ), b);
    c = s * τ;
    ρ = b / s;
  } else {
    auto const τ = b / a;
    c = std::copysign(1.f / std::sqrt(1.f + τ * τ), a);
    s = c * τ;
    ρ = a / c;
  }
  return std::make_tuple(c, s, ρ);
}

inline auto Rotation(float const a, float const b)
{
  float const ρ = std::hypot(a, b);
  float const c = a / ρ;
  float const s = b / ρ;
  return std::make_tuple(c, s, ρ);
}

template <typename OpPtr, typename PrePtr, typename RegPtr, typename Input, typename Output, typename Device>
inline void BidiagInit(
  OpPtr op,
  PrePtr M,
  Output &Mu,
  Output &u,
  Input &uλ,
  Input &v,
  float &α,
  float &β,
  float const λ,
  RegPtr opλ,
  Input &x,
  Eigen::TensorMap<Output const> const &b,
  Input const &x0,
  Input const &bλ,
  Device &dev)
{
  if (x0.size()) {
    CheckDimsEqual(x0.dimensions(), v.dimensions());
    x.device(dev) = x0;
    Mu.device(dev) = b - op->forward(x);
  } else {
    x.setZero();
    Mu.device(dev) = b;
  }
  (*M)(Mu, u);
  if (uλ.size()) {
    CheckDimsEqual(bλ.dimensions(), v.dimensions());
    uλ.device(dev) = (bλ - opλ->forward(x)) * x.constant(sqrt(λ));
    β = std::sqrt(CheckedDot(Mu, u) + CheckedDot(uλ, uλ));
  } else {
    β = std::sqrt(CheckedDot(Mu, u));
  }
  Mu.device(dev) = Mu / Mu.constant(β);
  u.device(dev) = u / u.constant(β);
  if (uλ.size()) {
    uλ.device(dev) = uλ / uλ.constant(β);
    v.device(dev) = op->adjoint(u) + (sqrt(λ) * opλ->adjoint(uλ));
  } else {
    v.device(dev) = op->adjoint(u);
  }
  α = std::sqrt(CheckedDot(v, v));
  v.device(dev) = v / v.constant(α);
}

template <typename OpPtr, typename PrePtr, typename RegPtr, typename Input, typename Output, typename Device>
inline void
Bidiag(OpPtr op, PrePtr M, Output &Mu, Output &u, Input &uλ, Input &v, float &α, float &β, float const λ, RegPtr opλ, Device &dev)
{
  Mu.device(dev) = op->forward(v) - α * Mu;
  (*M)(Mu, u);
  if (uλ.size()) {
    uλ.device(dev) = (std::sqrt(λ) * opλ->forward(v)) - (α * uλ);
    β = std::sqrt(CheckedDot(Mu, u) + CheckedDot(uλ, uλ));
  } else {
    β = std::sqrt(CheckedDot(Mu, u));
  }
  Mu.device(dev) = Mu / Mu.constant(β);
  u.device(dev) = u / u.constant(β);
  if (uλ.size()) {
    uλ.device(dev) = uλ / uλ.constant(β);
    v.device(dev) = op->adjoint(u) + (sqrt(λ) * opλ->adjoint(uλ)) - (β * v);
  } else {
    v.device(dev) = op->adjoint(u) - (β * v);
  }
  α = std::sqrt(CheckedDot(v, v));
  v.device(dev) = v / v.constant(α);
}

} // namespace

} // namespace rl
