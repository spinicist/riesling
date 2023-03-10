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

template <typename OpPtr, typename PrePtr, typename Reg, typename Input, typename Output, typename Device>
inline void BidiagInit(
  OpPtr op,
  PrePtr M,
  Output &Mu,
  Output &u,
  Input &v,
  float &α,
  float &β,
  float const λ,
  std::shared_ptr<Reg> opλ,
  typename Reg::Output &uλ,
  Input &x,
  Eigen::TensorMap<Output const> const &b,
  Input const &x0,
  typename Reg::Output const &bλ,
  Device &dev)
{
  if (x0.size()) {
    CheckDimsEqual(x0.dimensions(), v.dimensions());
    x.device(dev) = x0;
    Mu.device(dev) = b - op->cforward(x);
  } else {
    x.setZero();
    Mu.device(dev) = b;
  }
  u = M->cadjoint(Mu);
  if (uλ.size()) {
    CheckDimsEqual(bλ.dimensions(), uλ.dimensions());
    CheckDimsEqual(opλ->outputDimensions(), uλ.dimensions());
    uλ.device(dev) = std::sqrt(λ) * (bλ - opλ->cforward(x));
    β = std::sqrt(CheckedDot(Mu, u) + CheckedDot(uλ, uλ));
  } else {
    β = std::sqrt(CheckedDot(Mu, u));
  }
  Mu.device(dev) = Mu / Mu.constant(β);
  u.device(dev) = u / u.constant(β);
  if (uλ.size()) {
    uλ.device(dev) = uλ / uλ.constant(β);
    v.device(dev) = op->cadjoint(u) + std::sqrt(λ) * opλ->cadjoint(uλ);
  } else {
    v.device(dev) = op->cadjoint(u);
  }
  α = std::sqrt(CheckedDot(v, v));
  v.device(dev) = v / v.constant(α);
}

template <typename Op, typename Pre, typename Reg, typename Device>
inline void Bidiag(
  std::shared_ptr<Op> const op,
  std::shared_ptr<Pre> const M,
  typename Op::Output &Mu,
  typename Op::Output &u,
  typename Op::Input &v,
  float &α,
  float &β,
  float const λ,
  std::shared_ptr<Reg> const opλ,
  typename Reg::Output &uλ,
  Device &dev)
{
  Mu.device(dev) = op->cforward(v) - α * Mu;
  u = M->cadjoint(Mu);
  if (uλ.size()) {
    uλ.device(dev) = (std::sqrt(λ) * opλ->cforward(v)) - (α * uλ);
    β = std::sqrt(CheckedDot(Mu, u) + CheckedDot(uλ, uλ));
  } else {
    β = std::sqrt(CheckedDot(Mu, u));
  }
  Mu.device(dev) = Mu / Mu.constant(β);
  u.device(dev) = u / u.constant(β);
  if (uλ.size()) {
    uλ.device(dev) = uλ / uλ.constant(β);
    v.device(dev) = (op->cadjoint(u) + (std::sqrt(λ) * opλ->cadjoint(uλ))) - (β * v);
  } else {
    v.device(dev) = op->cadjoint(u) - (β * v);
  }
  α = std::sqrt(CheckedDot(v, v));
  v.device(dev) = v / v.constant(α);
}

} // namespace

} // namespace rl
