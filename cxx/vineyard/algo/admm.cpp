#include "admm.hpp"

#include "log.hpp"
#include "lsmr.hpp"
#include "op/top.hpp"
#include "signals.hpp"
#include "tensors.hpp"

namespace rl {

auto ADMM::run(Vector const &b, float const ρ) const -> Vector {
  return run(CMap{b.data(), b.rows()}, ρ);
}

auto ADMM::run(CMap const b, float ρ) const -> Vector
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

    z_i = prox_λ/ρ(F_i * x + u_{i-1})
    u_i = F_i * x + u_{i-1} - z_i
    */
  if (b.rows() != A->rows()) { Log::Fail("ADMM: b was size {} expected {}", b.rows(), A->rows()); }
  auto const dev = Threads::GlobalDevice();

  Index const                                      R = regs.size();
  std::vector<Vector>                              z(R), u(R);
  std::vector<std::shared_ptr<Ops::DiagScale<Cx>>> ρdiags(R);
  std::vector<std::shared_ptr<Ops::Op<Cx>>>        scaled_ops(R);
  for (Index ir = 0; ir < R; ir++) {
    Index const sz = regs[ir].T->rows();
    z[ir].resize(sz);
    z[ir].setZero();
    u[ir].resize(sz);
    u[ir].setZero();
    ρdiags[ir] = std::make_shared<Ops::DiagScale<Cx>>(sz, std::sqrt(ρ));
    scaled_ops[ir] = std::make_shared<Ops::Multiply<Cx>>(ρdiags[ir], regs[ir].T);
  }

  std::shared_ptr<Op> reg = std::make_shared<Ops::VStack<Cx>>(scaled_ops);
  std::shared_ptr<Op> Aʹ = std::make_shared<Ops::VStack<Cx>>(A, reg);
  std::shared_ptr<Op> I = std::make_shared<Ops::Identity<Cx>>(reg->rows());
  std::shared_ptr<Op> Mʹ = std::make_shared<Ops::DStack<Cx>>(M, I);

  LSMR lsmr{Aʹ, Mʹ, iters0, aTol, bTol, cTol};

  Vector x(A->cols());
  x.setZero();

  Vector bʹ(Aʹ->rows());
  bʹ.setZero();
  bʹ.head(A->rows()).device(dev) = b;

  Log::Print("ADMM Abs ε {}", ε);
  PushInterrupt();
  for (Index io = 0; io < outerLimit; io++) {
    Index start = A->rows();
    for (Index ir = 0; ir < R; ir++) {
      Index rr = regs[ir].T->rows();
      bʹ.segment(start, rr).device(dev) = std::sqrt(ρ) * (z[ir] - u[ir]);
      start += rr;
      ρdiags[ir]->scale = std::sqrt(ρ);
    }
    x = lsmr.run(bʹ, 0.f, x);
    lsmr.iterLimit = iters1;
    if (debug_x) { debug_x(io, x); }

    float normFx = 0.f, normz = 0.f, normu = 0.f, pRes = 0.f, dRes = 0.f;
    for (Index ir = 0; ir < R; ir++) {
      Vector const Fx = regs[ir].T->forward(x);
      Vector const Fxpu = Fx + u[ir];
      Vector const zprev = z[ir];
      regs[ir].P->apply(1.f / ρ, Fxpu, z[ir]);
      u[ir].device(dev) = Fxpu - z[ir];
      if (debug_z) { debug_z(io, ir, Fx, z[ir], u[ir]); }
      float const nFx = Fx.stableNorm();
      float const nz = z[ir].stableNorm();
      float const nu = regs[ir].T->adjoint(u[ir]).stableNorm();
      float const nP = (Fx - z[ir]).stableNorm();
      float const nD = (regs[ir].T->adjoint(z[ir] - zprev)).stableNorm();
      normFx += nFx * nFx;
      normz += nz * nz;
      normu += nu * nu;
      pRes += nP * nP;
      dRes += nD * nD;
      Log::Print("Reg {:02d} |Fx| {:4.3E} |z| {:4.3E} |F'u| {:4.3E}", ir, nFx, nz, nu);
    }
    float const normx = x.stableNorm();
    normFx = std::sqrt(normFx);
    normz = std::sqrt(normz);
    normu = std::sqrt(normu);
    pRes = std::sqrt(pRes) / std::max(normFx, normz);
    dRes = std::sqrt(dRes) / normu;

    Log::Print("ADMM {:02d} |x| {:4.3E} |Fx| {:4.3E} |z| {:4.3E} |F'u| {:4.3E} ρ {:4.3E} |Primal| {:4.3E} |Dual| {:4.3E}", io,
               normx, normFx, normz, normu, ρ, pRes, dRes);

    if ((pRes < ε) && (dRes < ε)) {
      Log::Print("Primal and dual tolerances achieved, stopping");
      break;
    }
    if (io > 0) {
      // ADMM Penalty Parameter Selection by Residual Balancing, Wohlberg 2017
      float const ratio = std::sqrt(pRes / dRes);
      float const τ = (ratio < 1.f) ? std::max(1.f / τmax, 1.f / ratio) : std::min(τmax, ratio);
      if (pRes > μ * dRes) {
        ρ *= τ;
        for (Index ir = 0; ir < R; ir++) {
          u[ir] /= τ;
        }
      } else if (dRes > μ * pRes) {
        ρ /= τ;
        for (Index ir = 0; ir < R; ir++) {
          u[ir] *= τ;
        }
      }
    }
    if (InterruptReceived()) { break; }
  }
  PopInterrupt();
  return x;
}

} // namespace rl
