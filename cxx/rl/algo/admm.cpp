#include "admm.hpp"

#include "../log/log.hpp"
#include "../op/top.hpp"
#include "../tensors.hpp"
#include "common.hpp"
#include "iter.hpp"
#include "lsmr.hpp"

namespace rl {

ADMM::ADMM(Op::Ptr AA, Op::Ptr MMinv, std::vector<Regularizer> const &r, Opts o, DebugX dx, DebugZ dz)
  : A{AA}
  , Minv{MMinv}
  , regs{r}
  , opts{o}
  , debug_x{dx}
  , debug_z{dz}
  , ρ(regs.size(), opts.ρ)
  , ρops(regs.size())
  , z(regs.size())
  , u(regs.size())
{
  Index const R = regs.size();
  Index       totalRows = 0;
  for (Index ir = 0; ir < R; ir++) {
    Index const sz = regs[ir].T ? regs[ir].T->rows() : A->cols();
    z[ir].resize(sz);
    z[ir].setZero();
    u[ir].resize(sz);
    u[ir].setZero();
    ρops[ir] = Ops::DiagScale::Make(regs[ir].T ? regs[ir].T : Ops::Identity::Make(sz), std::sqrt(ρ[ir]));
    totalRows += ρops[ir]->rows();
  }

  Aʹ = Ops::VStack::Make(A, ρops);
  Minvʹ = Minv ? Ops::DStack::Make(Minv, Ops::Identity::Make(totalRows)) : nullptr;
}

void ADMM::x_update(Index const io, Vector &x, Vector &bʹ) const
{
  auto const dev = Threads::CoreDevice();
  LSMR       lsmr{Aʹ, Minvʹ, nullptr, LSMR::Opts{io > 0 ? opts.iters1 : opts.iters0, opts.aTol, opts.bTol, opts.cTol}};

  Index start = A->rows();
  for (Index ir = 0; ir < regs.size(); ir++) {
    Index rr = regs[ir].T ? regs[ir].T->rows() : A->cols();
    bʹ.segment(start, rr).device(dev) = std::sqrt(ρ[ir]) * (z[ir] - u[ir]);
    start += rr;
    ρops[ir]->scale = std::sqrt(ρ[ir]);
  }
  x = lsmr.run(bʹ, x);
  if (debug_x) { debug_x(io, x); }
}

void ADMM::zu_update(Index const ir, Vector const &x, Vector const &zp) const
{
  // Note that in the Boyd primer relaxation is defined as ɑ * Ax - (1.f - ɑ) * (Bz - c) but this comes from the
  // constraint that Ax + Bz = c Our constraint is Fx = z, which defines A = F and B = -I, and hence the minus sign
  // becomes a plus in this line below, which matches the code examples on Boyd's website
  auto dev = Threads::CoreDevice();
  if (opts.ɑ > 0.f) {
    u[ir].device(dev) += opts.ɑ * x + (1.f - opts.ɑ) * zp;
  } else {
    u[ir].device(dev) += x;
  }
  z[ir].device(dev) = u[ir];
  regs[ir].P->apply(1.f / ρ[ir], z[ir]);
  u[ir].device(dev) = u[ir] - z[ir];
}

void ADMM::ρ_balance(Index const ir, float const pRes, float const dRes) const
{
  // ADMM Penalty Parameter Selection by Residual Balancing, Wohlberg 2017
  auto        dev = Threads::CoreDevice();
  float const ratio = std::sqrt(pRes / dRes);
  float const τ = (ratio < 1.f) ? std::max(1.f / opts.τmax, 1.f / ratio) : std::min(opts.τmax, ratio);
  if (pRes > opts.μ * dRes) {
    ρ[ir] *= τ;
    u[ir].device(dev) = u[ir] / τ;
  } else if (dRes > opts.μ * pRes) {
    ρ[ir] /= τ;
    u[ir].device(dev) = u[ir] * τ;
  }
}

auto ADMM::run(Vector const &b) const -> Vector { return run(CMap{b.data(), b.rows()}); }

auto ADMM::run(CMap b) const -> Vector
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
  if (b.rows() != A->rows()) { throw Log::Failure("ADMM", "b was size {} expected {}", b.rows(), A->rows()); }
  auto const dev = Threads::CoreDevice();

  Vector bʹ(Aʹ->rows());
  bʹ.setZero();
  bʹ.head(A->rows()).device(dev) = b;

  Vector x(A->cols());
  x.setZero();

  Log::Print("ADMM", "Max its {}/{}/{} Abs ε {}", opts.iters0, opts.iters1, opts.outerLimit, opts.ε);
  Iterating::Starting();
  for (Index io = 0; io < opts.outerLimit; io++) {
    x_update(io, x, bʹ);

    bool tolerance_reached = true;
    for (Index ir = 0; ir < regs.size(); ir++) {
      float  nz, nFx, nu, pRes, dRes;
      Vector zprev(z[ir].size());
      zprev.device(dev) = z[ir];

      if (regs[ir].T) {
        Vector Fx(u[ir].size());
        regs[ir].T->forward(x, Fx, std::sqrt(ρ[ir]));
        nFx = ParallelNorm(Fx);
        zu_update(ir, Fx, zprev);
        if (debug_z) { debug_z(io, ir, Fx, z[ir], u[ir]); }
        nu = ParallelNorm(regs[ir].T->adjoint(u[ir]));
        pRes = ParallelNorm(Fx - z[ir]);
        dRes = ParallelNorm(regs[ir].T->adjoint(z[ir] - zprev));
      } else {
        nFx = ParallelNorm(x);
        zu_update(ir, x, zprev);
        if (debug_z) { debug_z(io, ir, x, z[ir], u[ir]); }
        nu = ParallelNorm(u[ir]);
        pRes = ParallelNorm(x - z[ir]);
        dRes = ParallelNorm(z[ir] - zprev);
      }
      nz = ParallelNorm(z[ir]);
      Log::Print("ADMM", "{:02d}/{:02d} |Fx| {:3.2E} |z| {:3.2E} |F'u| {:3.2E} |Pres| {:3.2E} |Dres| {:3.2E} ρ {:3.2E}", io, ir,
                 nFx, nz, nu, pRes, dRes, ρ[ir]);
      if (pRes < std::numeric_limits<float>::min()) {
        Log::Print("ADMM", "Primal residual was zero, stopping");
        break;
      }
      if (pRes > opts.ε || dRes > opts.ε) { tolerance_reached = false; }
      if (opts.balance && io) { ρ_balance(ir, pRes, dRes); }
    }
    if (tolerance_reached) {
      Log::Print("ADMM", "Primal and dual residual tolerance achieved, stopping");
      break;
    }
    if (Iterating::ShouldStop("ADMM")) { break; }
  }
  Iterating::Finished();
  return x;
}

} // namespace rl
