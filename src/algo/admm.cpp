#include "admm.hpp"

#include "log.hpp"
#include "lsmr.hpp"
#include "op/tensorop.hpp"
#include "signals.hpp"
#include "tensorOps.hpp"

namespace rl {

auto ADMM::run(Cx *bdata, float ρ) const -> Vector
{
  /* See https://web.stanford.edu/~boyd/papers/admm/lasso/lasso_lsqr.html
   * For the least squares part we are solving:
  A'x = b'
  [     A            [     b
    √ρ_1 F_1           z_1 - u_1
    √ρ_2 F_2     x =   z_2 - u_2
      ...              ...
    √ρ_n F_n ]         z_n - u_n ]

    Then for the regularizers, in turn we do:

    z_i = prox_λ/ρ_i(F_i * x + u_i)
    u_i = F_i * x + u_i - z_i
    */

  Index const R = reg_ops.size();
  std::vector<Vector> z(R), zold(R), u(R), Fx(R), Fxpu(R);
  std::vector<std::shared_ptr<Op>> scaled_ops(R);
  for (Index ir = 0; ir < R; ir++) {
    z[ir].resize(reg_ops[ir]->rows());
    z[ir].setZero();
    zold[ir].resize(reg_ops[ir]->rows());
    zold[ir].setZero();
    u[ir].resize(reg_ops[ir]->rows());
    u[ir].setZero();
    Fx[ir].resize(reg_ops[ir]->rows());
    Fx[ir].setZero();
    Fxpu[ir].resize(reg_ops[ir]->rows());
    Fxpu[ir].setZero();
    scaled_ops[ir] = std::make_shared<LinOps::Scale<Cx>>(reg_ops[ir], std::sqrt(ρ));
  }

  std::shared_ptr<Op> reg = std::make_shared<LinOps::VStack<Cx>>(scaled_ops);
  std::shared_ptr<Op> Aʹ = std::make_shared<LinOps::VStack<Cx>>(A, reg);
  std::shared_ptr<Op> I = std::make_shared<LinOps::Identity<Cx>>(reg->rows());
  std::shared_ptr<Op> Mʹ = std::make_shared<LinOps::DStack<Cx>>(M, I);

  LSMR lsmr{Aʹ, Mʹ, lsqLimit, aTol, bTol, cTol, false};

  Vector x(A->cols());
  x.setZero();
  Map const b(bdata, A->rows());

  Vector bʹ(Aʹ->rows());
  bʹ.setZero();
  bʹ.head(A->rows()) = b;

  float const sqrtM = std::sqrt(x.rows());
  float const sqrtN = std::sqrt(bʹ.rows());
  Log::Print("ADMM Abs Tol {} Rel Tol {}", abstol, reltol);
  PushInterrupt();
  for (Index io = 0; io < outerLimit; io++) {
    Index start = A->rows();
    for (Index ir = 0; ir < R; ir++) {
      Index rr = reg_ops[ir]->rows();
      bʹ.segment(start, rr) = std::sqrt(ρ) * (z[ir] - u[ir]);
      start += rr;
      std::dynamic_pointer_cast<LinOps::Scale<Cx>>(scaled_ops[ir])->scale = std::sqrt(ρ);
    }
    x = lsmr.run(bʹ.data());
    if (auto At = std::dynamic_pointer_cast<TensorOperator<Cx, 4, 4>>(A)) {
      Log::Tensor(fmt::format("admm-{:02}-x", io), At->ishape, x.data());
    }

    float const normx = x.norm();
    float pNorm = 0.f, dNorm = 0.f, normz = 0.f, normu = 0.f;
    for (Index ir = 0; ir < R; ir++) {
      Fx[ir] = reg_ops[ir]->forward(x);
      Fxpu[ir] = Fx[ir] * α + z[ir] * (1.f - α) + u[ir];
      zold[ir] = z[ir];
      prox[ir]->apply(1.f / ρ, Fxpu[ir], z[ir]);
      u[ir] = Fxpu[ir] - z[ir];

      if (auto t = std::dynamic_pointer_cast<TensorOperator<Cx, 4, 5>>(reg_ops[ir])) {
        Log::Tensor(fmt::format("admm-{:02}-{:02}-Fx", io, ir), t->oshape, Fx[ir].data());
        Log::Tensor(fmt::format("admm-{:02}-{:02}-z", io, ir), t->oshape, z[ir].data());
        Log::Tensor(fmt::format("admm-{:02}-{:02}-u", io, ir), t->oshape, u[ir].data());
      }

      pNorm += (Fx[ir] - z[ir]).squaredNorm();
      dNorm += ρ * (z[ir] - zold[ir]).squaredNorm();

      normz += z[ir].squaredNorm();
      normu += u[ir].squaredNorm();
    }
    pNorm = std::sqrt(pNorm);
    dNorm = std::sqrt(dNorm);
    normz = std::sqrt(normz);
    normu = std::sqrt(normu);
    float const pEps = abstol * sqrtM + reltol * std::max(normx, normz);
    float const dEps = abstol * sqrtN + reltol * std::sqrt(R) * ρ * normu;
    Log::Print(
      FMT_STRING("ADMM Iter {:02d} ρ {} Primal || {:5.3E} ε {:5.3E} Dual || {:5.3E} ε {:5.3E} |z| "
                 "{:5.3E} |u| {:5.3E}"),
      io,
      ρ,
      pNorm,
      pEps,
      dNorm,
      dEps,
      normz,
      normu);

    if ((pNorm < pEps) || (dNorm < dEps)) {
      Log::Print("All primal and dual tolerances achieved, stopping");
      break;
    }
    if (io > 1) { // z_0 is zero, so perfectly reasonable dual residuals can trigger this
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
    if (InterruptReceived()) {
      break;
    }
  }
  PopInterrupt();
  return x;
}

} // namespace rl
