#include "admm.hpp"

#include "log.hpp"
#include "lsmr.hpp"
#include "op/tensorop.hpp"
#include "signals.hpp"
#include "tensorOps.hpp"

namespace rl {

auto ADMM::run(Cx const *bdata, float ρ) const -> Vector
{
  /* See https://web.stanford.edu/~boyd/papers/admm/lasso/lasso_lsqr.html
   * For the least squares part we are solving:
  A'x = b'
  [     A            [     b
    √ρ F_1             √ρ (z_1 - u_1)
    √ρ F_2     x =     √ρ (z_2 - u_2)
      ...              ...
    √ρ F_n ]           √ρ (z_n - u_n) ]

    Then for the regularizers, in turn we do:

    z_i = prox_λ/ρ(F_i * x + u_i)
    u_i = F_i * x + u_i - z_i
    */

  Index const R = reg_ops.size();
  std::vector<Vector> z(R), zprev(R), z0(R), Δz(R), u(R), u0(R), Δu(R), û(R), û0(R), Δû(R), Fx(R), Fx0(R), ΔFx(R), Fxpu(R);
  std::vector<std::shared_ptr<Ops::DiagScale<Cx>>> ρdiags(R);
  std::vector<std::shared_ptr<Ops::Op<Cx>>>        scaled_ops(R);
  for (Index ir = 0; ir < R; ir++) {
    Index const sz = reg_ops[ir]->rows();
    z[ir].resize(sz);
    z[ir].setZero();
    zprev[ir].resize(sz);
    zprev[ir].setZero();
    z0[ir].resize(sz);
    z0[ir].setZero();
    Δz[ir].resize(sz);
    Δz[ir].setZero();
    u[ir].resize(sz);
    u[ir].setZero();
    u0[ir].resize(sz);
    u0[ir].setZero();
    Δu[ir].resize(sz);
    Δu[ir].setZero();
    û[ir].resize(sz);
    û[ir].setZero();
    û0[ir].resize(sz);
    û0[ir].setZero();
    Δû[ir].resize(sz);
    Δû[ir].setZero();
    Fx[ir].resize(sz);
    Fx[ir].setZero();
    Fx0[ir].resize(sz);
    Fx0[ir].setZero();
    ΔFx[ir].resize(sz);
    ΔFx[ir].setZero();
    Fxpu[ir].resize(sz);
    Fxpu[ir].setZero();
    ρdiags[ir] = std::make_shared<Ops::DiagScale<Cx>>(sz, std::sqrt(ρ));
    scaled_ops[ir] = std::make_shared<Ops::Multiply<Cx>>(ρdiags[ir], reg_ops[ir]);
  }

  std::shared_ptr<Op> reg = std::make_shared<Ops::VStack<Cx>>(scaled_ops);
  std::shared_ptr<Op> Aʹ = std::make_shared<Ops::VStack<Cx>>(A, reg);
  std::shared_ptr<Op> I = std::make_shared<Ops::Identity<Cx>>(reg->rows());
  std::shared_ptr<Op> Mʹ = std::make_shared<Ops::DStack<Cx>>(M, I);

  LSMR lsmr{Aʹ, Mʹ, lsqLimit, aTol, bTol, cTol};

  Vector x(A->cols());
  x.setZero();
  CMap const b(bdata, A->rows());

  Vector bʹ(Aʹ->rows());
  bʹ.setZero();
  bʹ.head(A->rows()) = b;

  Log::Print("ADMM Abs ε {}", ε);
  PushInterrupt();
  for (Index io = 0; io < outerLimit; io++) {
    Index start = A->rows();
    for (Index ir = 0; ir < R; ir++) {
      Index rr = reg_ops[ir]->rows();
      bʹ.segment(start, rr) = std::sqrt(ρ) * (z[ir] - u[ir]);
      start += rr;
      ρdiags[ir]->scale = std::sqrt(ρ);
    }
    x = lsmr.run(bʹ.data(), 0.f, x.data());
    if (debug_x) { debug_x(io, x); }

    float const normx = x.norm();
    float       pNorm = 0.f, dNorm = 0.f, normz = 0.f, normu = 0.f;
    for (Index ir = 0; ir < R; ir++) {
      Fx[ir] = reg_ops[ir]->forward(x);
      Fxpu[ir] = Fx[ir] + u[ir];
      zprev[ir] = z[ir];
      prox[ir]->apply(1.f / ρ, Fxpu[ir], z[ir]);
      u[ir] = Fxpu[ir] - z[ir];

      pNorm += (Fx[ir] - z[ir]).squaredNorm();
      dNorm += ρ * (z[ir] - zprev[ir]).squaredNorm();

      normz += z[ir].squaredNorm();
      normu += u[ir].squaredNorm();

      if (debug_z) { debug_z(io, ir, z[ir]); }
    }
    pNorm = std::sqrt(pNorm);
    dNorm = std::sqrt(dNorm);
    normz = std::sqrt(normz);
    normu = std::sqrt(normu);
    float const pEps = ε * std::max(normx, normz);
    float const dEps = ε * ρ * normu;
    Log::Print(
      "ADMM Iter {:02d} |x| {:5.3E} |z| {:5.3E} |u| {:5.3E} ρ {} Primal || {:5.3E} ε {:5.3E} Dual || {:5.3E} ε {:5.3E}",
      io,
      normx,
      normz,
      normu,
      ρ,
      pNorm,
      pEps,
      dNorm,
      dEps);

    if ((pNorm < pEps) && (dNorm < dEps)) {
      Log::Print("Primal and dual tolerances achieved, stopping");
      break;
    }

    if (io % 2 == 1) {
      float Δû_dot_Δû = 0.f, ΔFx_dot_Δû = 0.f, ΔFx_dot_ΔFx = 0.f, Δu_dot_Δu = 0.f, Δz_dot_Δu = 0.f, Δz_dot_Δz = 0.f;
      float normΔFx = 0.f, normΔû = 0.f, normΔz = 0.f, normΔu = 0.f;
      for (Index ir = 0; ir < R; ir++) {
        û[ir] = Fxpu[ir] - zprev[ir];
        Δu[ir] = u[ir] - u0[ir];
        Δû[ir] = û[ir] - û0[ir];
        ΔFx[ir] = Fx[ir] - Fx0[ir];
        Δz[ir] = z[ir] - z0[ir];
        Δû_dot_Δû += std::real(Δû[ir].dot(Δû[ir]));
        ΔFx_dot_Δû += std::real(ΔFx[ir].dot(Δû[ir]));
        ΔFx_dot_ΔFx += std::real(ΔFx[ir].dot(ΔFx[ir]));
        Δu_dot_Δu += std::real(Δu[ir].dot(Δu[ir]));
        Δz_dot_Δu += std::real(Δz[ir].dot(Δu[ir]));
        Δz_dot_Δz += std::real(Δz[ir].dot(Δz[ir]));
        normΔFx += ΔFx[ir].squaredNorm();
        normΔû += Δû[ir].squaredNorm();
        normΔz += Δz[ir].squaredNorm();
        normΔu += Δu[ir].squaredNorm();
      }
      normΔFx = std::sqrt(normΔFx);
      normΔû = std::sqrt(normΔû);
      normΔz = std::sqrt(normΔz);
      normΔu = std::sqrt(normΔu);
      float const α̂sd = Δû_dot_Δû / ΔFx_dot_Δû;
      float const α̂mg = ΔFx_dot_Δû / ΔFx_dot_ΔFx;
      float const β̂sd = Δu_dot_Δu / Δz_dot_Δu;
      float const β̂mg = Δz_dot_Δu / Δz_dot_Δz;

      float const α̂ = (2.f * α̂mg > α̂sd) ? α̂mg : α̂sd - α̂mg / 2.f;
      float const β̂ = (2.f * β̂mg > β̂sd) ? β̂mg : β̂sd - β̂mg / 2.f;
      float const εcor = 0.2f;
      float const α̂cor = ΔFx_dot_Δû / (normΔFx * normΔû);
      float const β̂cor = Δz_dot_Δu / (normΔz * normΔu);

      if (α̂cor > εcor && β̂cor > εcor) {
        ρ = std::sqrt(α̂ * β̂);
      } else if (α̂cor > εcor && β̂cor <= εcor) {
        ρ = α̂;
      } else if (α̂cor <= εcor && β̂cor > εcor) {
        ρ = β̂;
      }
      Log::Print("Update ρ {} α̂cor {} β̂cor {} α̂ {} β̂  {}", ρ, α̂cor, β̂cor, α̂, β̂);
      // else ρ = ρ

      for (Index ir = 0; ir < R; ir++) {
        u0[ir] = u[ir];
        û0[ir] = û[ir];
        Fx0[ir] = Fx[ir];
        z0[ir] = z[ir];
      }
    }
    if (InterruptReceived()) { break; }
  }
  PopInterrupt();
  return x;
}

} // namespace rl
