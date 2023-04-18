#pragma once

#include "common.hpp"
#include "op/operator.hpp"

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

template <typename Scalar>
inline void BidiagInit(
  std::shared_ptr<LinOps::Op<Scalar>> op,
  std::shared_ptr<LinOps::Op<Scalar>> M,
  Eigen::Vector<Scalar, Eigen::Dynamic> &Mu,
  Eigen::Vector<Scalar, Eigen::Dynamic> &u,
  Eigen::Vector<Scalar, Eigen::Dynamic> &v,
  float &α,
  float &β,
  Eigen::Vector<Scalar, Eigen::Dynamic> &x,
  Eigen::Map<Eigen::Vector<Scalar, Eigen::Dynamic>> const &b,
  Scalar *x0)
{
  if (x0) {
    Eigen::Map<Eigen::Vector<Scalar, Eigen::Dynamic> const> xx0(x0, op->cols());
    x = xx0;
    Mu = b - op->forward(x);
  } else {
    x.setZero();
    Mu = b;
  }
  M->adjoint(Mu, u);
  β = std::sqrt(CheckedDot(Mu, u));
  Mu = Mu / β;
  u = u / β;
  op->adjoint(u, v);
  α = std::sqrt(CheckedDot(v, v));
  v = v / α;
}

template <typename Scalar>
inline void Bidiag(
  std::shared_ptr<LinOps::Op<Scalar>> const op,
  std::shared_ptr<LinOps::Op<Scalar>> const M,
  Eigen::Vector<Scalar, Eigen::Dynamic> &Mu,
  Eigen::Vector<Scalar, Eigen::Dynamic> &u,
  Eigen::Vector<Scalar, Eigen::Dynamic> &v,
  float &α,
  float &β)
{
  Mu = op->forward(v) - α * Mu;
  M->adjoint(Mu, u);
  β = std::sqrt(CheckedDot(Mu, u));
  Mu = Mu / β;
  u = u / β;
  v = op->adjoint(u) - (β * v);
  α = std::sqrt(CheckedDot(v, v));
  v = v / α;
}

} // namespace

} // namespace rl
