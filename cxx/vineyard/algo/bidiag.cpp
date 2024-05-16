#include "bidiag.hpp"

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

void BidiagInit(std::shared_ptr<Ops::Op<Cx>>           op,
                std::shared_ptr<Ops::Op<Cx>>           M,
                Eigen::VectorXcf                      &Mu,
                Eigen::VectorXcf                      &u,
                Eigen::VectorXcf                      &v,
                float                                 &α,
                float                                 &β,
                Eigen::VectorXcf                      &x,
                Eigen::VectorXcf::ConstAlignedMapType &b,
                Cx                                    *x0)
{
  if (x0) {
    Eigen::Map<Eigen::VectorXcf const> xx0(x0, op->cols());
    x = xx0;
    // Reuse u to save space
    op->forward(x, u);
    Mu.device(Threads::GlobalDevice()) = b - u;
  } else {
    x.setZero();
    Mu.device(Threads::GlobalDevice()) = b;
  }
  M->forward(Mu, u);
  β = std::sqrt(CheckedDot(Mu, u));
  Mu /= β;
  u /= β;
  op->adjoint(u, v);
  α = std::sqrt(CheckedDot(v, v));
  v /= α;
}

void Bidiag(std::shared_ptr<Ops::Op<Cx>> const op,
            std::shared_ptr<Ops::Op<Cx>> const M,
            Eigen::VectorXcf                  &Mu,
            Eigen::VectorXcf                  &u,
            Eigen::VectorXcf                  &v,
            float                             &α,
            float                             &β)
{
  // Re-use u to save space
  op->forward(v, u);
  Mu.device(Threads::GlobalDevice()) = u - α * Mu;
  M->forward(Mu, u);
  β = std::sqrt(CheckedDot(Mu, u));
  Mu.device(Threads::GlobalDevice()) = Mu / β;
  u.device(Threads::GlobalDevice()) = u / β;
  v = op->adjoint(u) - (β * v);
  α = std::sqrt(CheckedDot(v, v));
  v.device(Threads::GlobalDevice()) = v / α;
}

} // namespace rl
