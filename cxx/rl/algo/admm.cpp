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
  , z(regs.size())
  , u(regs.size())
  , ρscalers(regs.size())
{
  Index const R = regs.size();
  Index       totalRows = 0;
  for (Index ir = 0; ir < R; ir++) {
    Index const sz = regs[ir].T ? regs[ir].T->rows() : A->cols();
    z[ir].resize(sz);
    z[ir].setZero();
    u[ir].resize(sz);
    u[ir].setZero();
    ρscalers[ir] = Ops::DiagScale::Make(regs[ir].T ? regs[ir].T : Ops::Identity::Make(sz), std::sqrt(opts.ρ));
    totalRows += ρscalers[ir]->rows();
  }

  Aʹ = Ops::VStack::Make(A, ρscalers);
  Minvʹ = Minv ? Ops::DStack::Make(Minv, Ops::Identity::Make(totalRows)) : nullptr;
}

void ADMM::zu_update(Vector const &x, Vector const &zp, Index const ir, float const ρ) const
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
  regs[ir].P->apply(1.f / ρ, z[ir]);
  u[ir].device(dev) = u[ir] - z[ir];
}

auto ADMM::ρ_balance(float ρ, float const pRes, float const dRes) const -> float
{
  // ADMM Penalty Parameter Selection by Residual Balancing, Wohlberg 2017
  auto        dev = Threads::CoreDevice();
  float const ratio = std::sqrt(pRes / dRes);
  float const τ = (ratio < 1.f) ? std::max(1.f / opts.τmax, 1.f / ratio) : std::min(opts.τmax, ratio);
  if (pRes > opts.μ * dRes) {
    ρ *= τ;
    for (Index ir = 0; ir < regs.size(); ir++) {
      u[ir].device(dev) = u[ir] / τ;
    }
  } else if (dRes > opts.μ * pRes) {
    ρ /= τ;
    for (Index ir = 0; ir < regs.size(); ir++) {
      u[ir].device(dev) = u[ir] * τ;
    }
  }
  return ρ;
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
  float      ρ = opts.ρ;
  LSMR       lsmr{Aʹ, Minvʹ, nullptr, LSMR::Opts{opts.iters0, opts.aTol, opts.bTol, opts.cTol}};

  Vector x(A->cols());
  x.setZero();

  Vector bʹ(Aʹ->rows());
  bʹ.setZero();
  bʹ.head(A->rows()).device(dev) = b;

  Log::Print("ADMM", "Max its {}/{}/{} Abs ε {}", opts.iters0, opts.iters1, opts.outerLimit, opts.ε);
  Iterating::Starting();
  for (Index io = 0; io < opts.outerLimit; io++) {
    Index start = A->rows();
    for (Index ir = 0; ir < regs.size(); ir++) {
      Index rr = regs[ir].T ? regs[ir].T->rows() : A->cols();
      bʹ.segment(start, rr).device(dev) = std::sqrt(ρ) * (z[ir] - u[ir]);
      start += rr;
      ρscalers[ir]->scale = std::sqrt(ρ);
    }
    x = lsmr.run(bʹ, x);
    lsmr.opts.imax = opts.iters1;
    if (debug_x) { debug_x(io, x); }

    float normFx = 0.f, normz = 0.f, normu = 0.f, pRes = 0.f, dRes = 0.f;
    for (Index ir = 0; ir < regs.size(); ir++) {
      float  nz, nFx, nu, nP, nD;
      Vector zprev(z[ir].size());
      zprev.device(dev) = z[ir];

      if (regs[ir].T) {
        Vector Fx(u[ir].size());
        regs[ir].T->forward(x, Fx, std::sqrt(ρ));
        nFx = ParallelNorm(Fx);
        zu_update(Fx, zprev, ir, ρ);
        if (debug_z) { debug_z(io, ir, Fx, z[ir], u[ir]); }
        nu = ParallelNorm(regs[ir].T->adjoint(u[ir]));
        nP = ParallelNorm(Fx - z[ir]);
        nD = ParallelNorm(regs[ir].T->adjoint(z[ir] - zprev));
      } else {
        nFx = ParallelNorm(x);
        zu_update(x, zprev, ir, ρ);
        if (debug_z) { debug_z(io, ir, x, z[ir], u[ir]); }
        nu = ParallelNorm(u[ir]);
        nP = ParallelNorm(x - z[ir]);
        nD = ParallelNorm(z[ir] - zprev);
      }
      nz = ParallelNorm(z[ir]);
      Log::Print("ADMM", "Reg {:02d} |Fx| {:3.2E} |z| {:3.2E} |F'u| {:3.2E}", ir, nFx, nz, nu);
      normz += nz * nz;
      normu += nu * nu;
      pRes += nP * nP;
      dRes += nD * nD;
    }
    float const normx = ParallelNorm(x);
    normz = std::sqrt(normz);
    normu = std::sqrt(normu);
    pRes = std::sqrt(pRes) / std::max(normFx, normz);
    dRes = std::sqrt(dRes) / normu;

    Log::Print("ADMM", "{:02d} |x| {:3.2E} |z| {:3.2E} |F'u| {:3.2E} ρ {:3.2E} |Pr| {:3.2E} |Du| {:3.2E}", io, normx, normz,
               normu, ρ, pRes, dRes);

    if (pRes < std::numeric_limits<float>::min()) {
      Log::Print("ADMM", "Primal residual was numerically zero, stopping");
      break;
    }
    if ((pRes < opts.ε) && (dRes < opts.ε)) {
      Log::Print("ADMM", "Primal and dual tolerances achieved, stopping");
      break;
    }
    if (opts.balance && io > 0) { ρ = ρ_balance(ρ, pRes, dRes); }
    if (Iterating::ShouldStop("ADMM")) { break; }
  }
  Iterating::Finished();
  return x;
}

} // namespace rl
