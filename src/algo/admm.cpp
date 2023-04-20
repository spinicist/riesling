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

  Index const N = reg_ops.size();
  std::vector<Vector> z(N), zold(N), u(N), Fx(N), Fxpu(N);
  std::vector<std::shared_ptr<Op>> scaled_ops(N);
  std::vector<float> ρs(N);
  for (Index ir = 0; ir < N; ir++) {
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
    ρs[ir] = ρ;
    scaled_ops[ir] = std::make_shared<LinOps::Scale<Cx>>(reg_ops[ir], std::sqrt(ρs[ir]));
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
  Log::Print(FMT_STRING("ADMM Abs Tol {} Rel Tol {}"), abstol, reltol);
  PushInterrupt();
  for (Index io = 0; io < outerLimit; io++) {
    Index start = A->rows();
    for (Index ir = 0; ir < N; ir++) {
      auto &ρr = ρs[ir];
      Index rr = reg_ops[ir]->rows();
      bʹ.segment(start, rr) = std::sqrt(ρr) * (z[ir] - u[ir]);
      start += rr;
      std::dynamic_pointer_cast<LinOps::Scale<Cx>>(scaled_ops[ir])->scale = std::sqrt(ρr);
    }
    x = lsmr.run(bʹ.data());
    float const normx = x.norm();

    bool converged = true;
    if (auto At = std::dynamic_pointer_cast<TensorOperator<Cx, 4, 4>>(A)) {
      Log::Tensor(fmt::format(FMT_STRING("admm-{:02}-x"), io), At->ishape, x.data());
    }

    for (Index ir = 0; ir < N; ir++) {
      auto &ρr = ρs[ir];
      Fx[ir] = reg_ops[ir]->forward(x);
      Fxpu[ir] = Fx[ir] * α + z[ir] * (1.f - α) + u[ir];
      zold[ir] = z[ir];
      prox[ir]->apply(1.f / ρr, Fxpu[ir], z[ir]);
      u[ir] = Fxpu[ir] - z[ir];

      if (auto t = std::dynamic_pointer_cast<TensorOperator<Cx, 4, 5>>(reg_ops[ir])) {
        Log::Tensor(fmt::format(FMT_STRING("admm-{:02}-{:02}-Fx"), io, ir), t->oshape, Fx[ir].data());
        Log::Tensor(fmt::format(FMT_STRING("admm-{:02}-{:02}-z"), io, ir), t->oshape, z[ir].data());
        Log::Tensor(fmt::format(FMT_STRING("admm-{:02}-{:02}-u"), io, ir), t->oshape, u[ir].data());
      }

      float const pNorm = (Fx[ir] - z[ir]).norm();
      float const dNorm = ρr * (z[ir] - zold[ir]).norm();

      float const normFx = Fx[ir].norm();
      float const normz = z[ir].norm();
      float const normu = u[ir].norm();

      float const pEps = abstol * sqrtM + reltol * std::max(normx, normz);
      float const dEps = abstol * sqrtN + reltol * ρr * normu;

      Log::Print(
        FMT_STRING("ADMM Iter {:02d}:{:02d} ρ {} Primal || {:5.3E} ε {:5.3E} Dual || {:5.3E} ε {:5.3E} |Fx| {:5.3E} |z| "
                   "{:5.3E} |u| {:5.3E}"),
        io,
        ir,
        ρr,
        pNorm,
        pEps,
        dNorm,
        dEps,
        normFx,
        normz,
        normu);

      if ((pNorm > pEps) || (dNorm > dEps)) {
        converged = false;
      }
      if (pNorm > μ * dNorm) {
        ρr *= τ;
        u[ir] /= τ;
      } else if (dNorm > μ * pNorm) {
        ρr /= τ;
        u[ir] *= τ;
      }
    }
    if (converged) {
      Log::Print("All primal and dual tolerances achieved, stopping");
    }

    if (InterruptReceived()) {
      break;
    }
  }
  PopInterrupt();
  return x;
}

} // namespace rl
