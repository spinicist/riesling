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

  Index const                                      R = reg_ops.size();
  std::vector<Vector>                              z(R), zold(R), u(R), Fx(R), Fxpu(R);
  std::vector<std::shared_ptr<Ops::DiagScale<Cx>>> ρdiags(R);
  std::vector<std::shared_ptr<Ops::Op<Cx>>>        scaled_ops(R);
  for (Index ir = 0; ir < R; ir++) {
    Index const sz = reg_ops[ir]->rows();
    z[ir].resize(sz);
    z[ir].setZero();
    zold[ir].resize(sz);
    zold[ir].setZero();
    u[ir].resize(sz);
    u[ir].setZero();
    Fx[ir].resize(sz);
    Fx[ir].setZero();
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
      zold[ir] = z[ir];
      prox[ir]->apply(1.f / ρ, Fxpu[ir], z[ir]);
      u[ir] = Fxpu[ir] - z[ir];

      pNorm += (Fx[ir] - z[ir]).squaredNorm();
      dNorm += ρ * (z[ir] - zold[ir]).squaredNorm();

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
    if (io > 0) { // z_0 is zero, so perfectly reasonable dual residuals can trigger this
      if (pNorm > μ * dNorm) {
        ρ *= τ;
        for (Index ir = 0; ir < R; ir++) {
          u[ir] /= τ;
        }
      } else if (dNorm > μ * pNorm) {
        ρ /= τ;
        for (Index ir = 0; ir < R; ir++) {
          u[ir] *= τ;
        }
      }
    }
    if (hogwild && !(io == 0) && !(io & (io - 1))) {
      ρ *= 2.f;
      for (Index ir = 0; ir < R; ir++) {
        u[ir] /= 2.f;
      }
    }
    if (InterruptReceived()) { break; }
  }
  PopInterrupt();
  return x;
}

} // namespace rl
