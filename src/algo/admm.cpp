#include "admm.hpp"

#include "log.hpp"
#include "lsmr.hpp"
#include "op/tensorop.hpp"
#include "signals.hpp"
#include "tensorOps.hpp"

namespace rl {

auto ADMM::run(Cx const *bdata, float const ρ) const -> Vector
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
  std::vector<Vector> z(R), zprev(R), z0(R), Δz(R), u(R), u0(R), Δu(R), û(R), û0(R), Δû(R), Fx(R), Fx0(R), ΔFx(R), Fxpu(R);
  std::vector<float>                               ρs(R);
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
    û[ir].resize(sz);
    û[ir].setZero();
    û0[ir].resize(sz);
    û0[ir].setZero();
    Δû[ir].resize(sz);
    Δû[ir].setZero();
    Fx[ir].resize(sz);
    Fx[ir].setZero();
    Fx0[ir].resize(sz);
    Fx0[ir].setZero();
    ΔFx[ir].resize(sz);
    ΔFx[ir].setZero();
    Fxpu[ir].resize(sz);
    Fxpu[ir].setZero();
    ρs[ir] = ρ;
    ρdiags[ir] = std::make_shared<Ops::DiagScale<Cx>>(sz, std::sqrt(ρs[ir]));
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

  Log::Print("ADMM ε {:4.3E}", ε);
  PushInterrupt();
  for (Index io = 0; io < outerLimit; io++) {
    Index start = A->rows();
    for (Index ir = 0; ir < R; ir++) {
      Index rr = reg_ops[ir]->rows();
      bʹ.segment(start, rr) = std::sqrt(ρs[ir]) * (z[ir] - u[ir]);
      start += rr;
      ρdiags[ir]->scale = std::sqrt(ρs[ir]);
    }
    x = lsmr.run(bʹ.data(), 0.f, x.data());
    float const normx = x.norm();
    Log::Print("ADMM Iter {:02d} |x| {:4.3E}", io, normx);
    if (debug_x) { debug_x(io, x); }

    bool converged = true;
    for (Index ir = 0; ir < R; ir++) {
      Fx[ir] = reg_ops[ir]->forward(x);
      Fxpu[ir] = Fx[ir] + u[ir];
      zprev[ir] = z[ir];
      prox[ir]->apply(1.f / ρs[ir], Fxpu[ir], z[ir]);
      u[ir] = Fxpu[ir] - z[ir];
      float const normFx = Fx[ir].norm();
      float const normz = z[ir].norm();
      float const normu = u[ir].norm();
      if (debug_z) { debug_z(io, ir, Fx[ir], u[ir], z[ir]); }
      // Relative residuals as per Wohlberg 2017
      float const pRes = (Fx[ir] - z[ir]).norm() / std::max(normx, normz);
      float const dRes = (z[ir] - zprev[ir]).norm() / normu;

      Log::Print("Reg {:02d} ρ {:4.3E} |Fx| {:4.3E} |z| {:4.3E} |u| {:4.3E} |Primal| {:4.3E} |Dual| {:4.3E}", //
                 ir, ρs[ir], normFx, normz, normu, pRes, dRes);
      if ((pRes > ε) || (dRes > ε)) { converged = false; }

      if (io % 2 == 1) {
        û[ir] = Fxpu[ir] - zprev[ir];
        Δu[ir] = u[ir] - u0[ir];
        Δû[ir] = û[ir] - û0[ir];
        ΔFx[ir] = Fx[ir] - Fx0[ir];
        Δz[ir] = z[ir] - z0[ir];
        Cx const    Δû_dot_Δû = Δû[ir].dot(Δû[ir]);
        Cx const    ΔFx_dot_Δû = ΔFx[ir].dot(Δû[ir]);
        Cx const    ΔFx_dot_ΔFx = ΔFx[ir].dot(ΔFx[ir]);
        Cx const    Δu_dot_Δu = Δu[ir].dot(Δu[ir]);
        Cx const    Δz_dot_Δu = Δz[ir].dot(Δu[ir]);
        Cx const    Δz_dot_Δz = Δz[ir].dot(Δz[ir]);
        float const normΔFx = ΔFx[ir].norm();
        float const normΔû = Δû[ir].norm();
        float const normΔz = Δz[ir].norm();
        float const normΔu = Δu[ir].norm();
        float const α̂sd = std::abs(Δû_dot_Δû / ΔFx_dot_Δû);
        float const α̂mg = std::abs(ΔFx_dot_Δû / ΔFx_dot_ΔFx);
        float const β̂sd = std::abs(Δu_dot_Δu / Δz_dot_Δu);
        float const β̂mg = std::abs(Δz_dot_Δu / Δz_dot_Δz);

        float const α̂ = (2.f * α̂mg > α̂sd) ? α̂mg : α̂sd - α̂mg / 2.f;
        float const β̂ = (2.f * β̂mg > β̂sd) ? β̂mg : β̂sd - β̂mg / 2.f;
        float const εcor = 0.2f;
        float const α̂cor = std::abs(ΔFx_dot_Δû) / (normΔFx * normΔû);
        float const β̂cor = std::abs(Δz_dot_Δu) / (normΔz * normΔu);

        float const ρold = ρs[ir];
        if (α̂cor > εcor && β̂cor > εcor) {
          ρs[ir] = std::sqrt(α̂ * β̂);
        } else if (α̂cor > εcor && β̂cor <= εcor) {
          ρs[ir] = α̂;
        } else if (α̂cor <= εcor && β̂cor > εcor) {
          ρs[ir] = β̂;
        }
        float const τ = ρold / ρs[ir];

        if (τ != 1.f) {
          u[ir] = u[ir] * τ;
          u0[ir] = u[ir];
          û0[ir] = û[ir];
          Fx0[ir] = Fx[ir];
          z0[ir] = z[ir];
        }
      }
    }
    if (converged) {
      Log::Print("All primal and dual tolerances achieved, stopping");
      break;
    }

    if (InterruptReceived()) { break; }
  }
  PopInterrupt();
  return x;
}

} // namespace rl
