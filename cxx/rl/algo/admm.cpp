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

auto ADMM::zu_update(Index const io, Index const ir, Vector const &Fx) const -> float
{
  // Note that in the Boyd primer relaxation is defined as ɑ * Ax - (1.f - ɑ) * (Bz - c) but this comes from the
  // constraint that Ax + Bz = c Our constraint is Fx = z, which defines A = F and B = -I, and hence the minus sign
  // becomes a plus in this line below, which matches the code examples on Boyd's website
	auto dev = Threads::CoreDevice();
  Vector z_k(z[ir].size());
  z_k.device(dev) = z[ir];
	Vector Fxpu(Fx.size());
	Fxpu.device(dev) = Fx + u[ir];
  z[ir].device(dev) = Fxpu;
  regs[ir].P->apply(1.f / ρ[ir], z[ir]);

  float const ρ_k = ρ[ir];
	float const p = ParallelNorm(ρ_k*(Fx - z[ir]));
	float const q = ParallelNorm(z[ir] - z_k);
	if (io % opts.T == (opts.T - 1)) {
		float const l = std::numeric_limits<float>::min();
		if (p < l && q < l) {
			ρ[ir] = ρ_k;
		} else if (p < l) {
			ρ[ir] = ρ_k / opts.τ;
		} else if (q < l) {
			ρ[ir] = ρ_k * opts.τ;
		} else {
			ρ[ir] = p / q;
		}
	}
  u[ir].device(dev) = (ρ_k / ρ[ir]) * (Fxpu - z[ir]);
  Log::Print("ADMM", "{:<4s} |Fx| {:3.2E} |z| {:3.2E} |u| {:3.2E} p {:3.2E} q {:3.2E} ρ {:3.2E}", //
             regs[ir].P->name, ParallelNorm(Fx), ParallelNorm(z[ir]), ParallelNorm(u[ir]), p, q, ρ[ir]);
  return (ρ_k / ρ[ir]) * p;
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
    Log::Print("ADMM", "{} |x| {:3.2E}", io, ParallelNorm(x));
    bool tolerance_reached = true;
    for (Index ir = 0; ir < regs.size(); ir++) {
      float pRes = 0.f;
      if (regs[ir].T) {
        Vector Fx(u[ir].size());
        regs[ir].T->forward(x, Fx, std::sqrt(ρ[ir]));
        pRes = zu_update(io, ir, Fx);
        if (debug_z) { debug_z(io, ir, Fx, z[ir], u[ir]); }
      } else {
        pRes = zu_update(io, ir, x);
        if (debug_z) { debug_z(io, ir, x, z[ir], u[ir]); }
      }
      if (pRes > opts.ε) { tolerance_reached = false; }
    }
    if (tolerance_reached) {
      Log::Print("ADMM", "Primal tolerance achieved, stopping");
      break;
    }
    if (Iterating::ShouldStop("ADMM")) { break; }
  }
  Iterating::Finished();
  return x;
}

} // namespace rl
