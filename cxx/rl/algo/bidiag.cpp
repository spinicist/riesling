#include "bidiag.hpp"

#include "../log/log.hpp"

namespace rl {

auto StableGivens(float const a, float const b) -> std::tuple<float, float, float>
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

auto Rotation(float const a, float const b) -> std::tuple<float, float, float>
{
  float const ρ = std::hypot(a, b);
  float const c = a / ρ;
  float const s = b / ρ;
  return std::make_tuple(c, s, ρ);
}

Bidiag::Bidiag(Ptr A_, Ptr M_, Ptr N_, Vector &x, CMap b, CMap x0)
  : A{A_}
  , M{M_}
  , N{N_}
{
  u.resize(A->rows());
  if (M) { Mu.resize(A->rows()); }
  v.resize(A->cols());
  if (N) { Nv.resize(A->cols()); }

  if (x0.size()) {
    x = x0;
    A->forward(x, u); // Reuse u to save space
    if (M) {
      Mu.device(Threads::CoreDevice()) = b - u;
    } else {
      u.device(Threads::CoreDevice()) = b - u;
    }
  } else {
    x.setZero();
    if (M) {
      Mu.device(Threads::CoreDevice()) = b;
    } else {
      u.device(Threads::CoreDevice()) = b;
    }
  }
  if (M) {
    M->inverse(Mu, u);
    β = std::sqrt(CheckedDot(Mu, u));
    Mu.device(Threads::CoreDevice()) = Mu * (1.f / β);
  } else {
    β = std::sqrt(CheckedDot(u, u));
  }
  u.device(Threads::CoreDevice()) = u * (1.f / β);
  if (N) {
    Nv = A->adjoint(u);
    N->inverse(Nv, v);
    α = std::sqrt(CheckedDot(v, v));
    Nv.device(Threads::CoreDevice()) = v * (1.f / α);
  } else {
    A->adjoint(u, v);
    α = std::sqrt(CheckedDot(v, v));
  }
  v.device(Threads::CoreDevice()) = v * (1.f / α);
}

void Bidiag::next()
{
  if (M) {
    Mu.device(Threads::CoreDevice()) = -α * Mu;
    A->iforward(v, Mu);
    M->inverse(Mu, u);
    β = std::sqrt(CheckedDot(Mu, u));
    Mu.device(Threads::CoreDevice()) = Mu / β;
  } else {
    u.device(Threads::CoreDevice()) = -α * u;
    A->iforward(v, u);
    β = std::sqrt(CheckedDot(u, u));
  }
  u.device(Threads::CoreDevice()) = u * (1.f / β);

  if (N) {
    Nv.device(Threads::CoreDevice()) = -β * Nv;
    A->iadjoint(u, Nv);
    N->inverse(Nv, v);
    α = std::sqrt(CheckedDot(Nv, v));
    Nv.device(Threads::CoreDevice()) = Nv * (1.f / α);
  } else {
    v.device(Threads::CoreDevice()) = -β * v;
    A->iadjoint(u, v);
    α = std::sqrt(CheckedDot(v, v));
  }
  v.device(Threads::CoreDevice()) = v * (1.f / α);
}

} // namespace rl
