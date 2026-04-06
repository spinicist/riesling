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
  , y(regs.size())
{
  Index const R = regs.size();
  Index       totalRows = 0;
  for (Index ir = 0; ir < R; ir++) {
    Index const sz = regs[ir].T ? regs[ir].T->rows() : A->cols();
    z[ir].resize(sz);
    z[ir].setZero();
    y[ir].resize(sz);
    y[ir].setZero();
    ρops[ir] = Ops::DiagScale::Make(regs[ir].T ? regs[ir].T : Ops::Identity::Make(sz), std::sqrt(ρ[ir]));
    totalRows += ρops[ir]->rows();
  }

  Aʹ = Ops::VStack::Make(A, ρops);
  Minvʹ = Minv ? Ops::DStack::Make(Minv, Ops::Identity::Make(totalRows)) : nullptr;
  bʹ.resize(Aʹ->rows());
  bʹ.setZero();
}

void ADMM::x_update(Index const io, Vector const& x_k, Vector& x) const
{
  /* See https://web.stanford.edu/~boyd/papers/admm/lasso/lasso_lsqr.html
   * The x-update solves an augmented Least-Squares problem.
   * We are now using the unscaled version of ADMM (variables x, z, y)
	 *  A'x = b'
	 *	A'  =  [     A    b' = [     b
	 *	       √ρ F_i          √ρ (z_i - y_i/ρ)
	 *	          ...              ...
	 *	       √ρ F_n ]        √ρ (z_n - y_n/ρ) ]
	 * This has normal equations (for one regularizer):
	 * (A'tA + ρFtF) x = (A'tb + ρFt(z - y/ρ))
   */

  auto const dev = Threads::CoreDevice();
  LSMR       lsmr{Aʹ, Minvʹ, nullptr, LSMR::Opts{io > 0 ? opts.iters1 : opts.iters0, opts.aTol, opts.bTol, opts.cTol}};

  Index start = A->rows();
  for (Index ir = 0; ir < regs.size(); ir++) {
    bʹ.segment(start, z[ir].rows()).device(dev) = std::sqrt(ρ[ir]) * (z[ir] - y[ir]/ρ[ir]);
    start += z[ir].rows();
    ρops[ir]->scale = std::sqrt(ρ[ir]);
  }
  x = lsmr.run(bʹ, x_k);
  if (debug_x) { debug_x(io, x); }
}

void ADMM::zy_update(Index const io, Index const ir, Vector const &Fx) const
{
	/*
    z_i = prox_λ/ρ(F_i * x + u_{i-1})
    y_i = y + F_i * x - z_i
	*/

  // Note that in the Boyd primer relaxation is defined as ɑ * Ax - (1.f - ɑ) * (Bz - c) but this comes from the
  // constraint that Ax + Bz = c Our constraint is Fx = z, which defines A = F and B = -I, and hence the minus sign
  // becomes a plus in this line below, which matches the code examples on Boyd's website
	auto dev = Threads::CoreDevice();
  Vector temp(Fx.size());

  // Some ridiculous memory re-use
	temp.device(dev) = Fx + y[ir]/ρ[ir];
	regs[ir].P->apply(1.f / ρ[ir], temp);
  z[ir].device(dev) = z[ir] - temp;
  float const q = ParallelNorm(z[ir]);
  z[ir].device(dev) = temp;

 	temp.device(dev) = Fx - z[ir];
 	float const p = ρ[ir] * ParallelNorm(temp);
  y[ir].device(dev) = y[ir] + ρ[ir] * temp;
  
  Log::Print("ADMM", "{:<4s} |Fx| {:3.2E} |z| {:3.2E} |y| {:3.2E} ρ {:3.2E} p {:3.2E} q {:3.2E}",
             regs[ir].P->name, ParallelNorm(Fx), ParallelNorm(z[ir]), ParallelNorm(y[ir]), ρ[ir], p, q);
  
	if (io % opts.T == (opts.T - 1)) {
		float const l = std::numeric_limits<float>::min();
		if (p < l && q < l) {
			// Do nothing
		} else if (p < l) {
			ρ[ir] /= opts.τ;
		} else if (q < l) {
			ρ[ir] *= opts.τ;
		} else {
			ρ[ir] = p / q;
		}
		Log::Print("ADMM", "New ρ {:3.2E}", ρ[ir]);
	}

	if (debug_z) { debug_z(io, ir, Fx, z[ir], y[ir]); }
}

auto ADMM::run(Vector const &b) const -> Vector { return run(CMap{b.data(), b.rows()}); }

auto ADMM::run(CMap b) const -> Vector
{
  if (b.rows() != A->rows()) { throw Log::Failure("ADMM", "b was size {} expected {}", b.rows(), A->rows()); }
  auto const dev = Threads::CoreDevice();

  bʹ.head(A->rows()).device(dev) = b;

  Vector x(A->cols()), x_k(A->cols());
  x.setZero();
  x_k.setZero();

  Log::Print("ADMM", "Max its {}/{}/{} Abs ε {}", opts.iters0, opts.iters1, opts.outerLimit, opts.ε);
  Iterating::Starting();
  for (Index io = 0; io < opts.outerLimit; io++) {
    x_update(io, x_k, x);
    float const nx = ParallelNorm(x);
		x_k.device(dev) = x - x_k;
		float const Δ = ParallelNorm(x_k) / nx;
    Log::Print("ADMM", "{} |x| {:3.2E} |Δ| {:3.2E}", io, nx, Δ);
    if (Δ < opts.ε) {
    	Log::Print("ADMM", "Convergence tolerance {:3.2E} reached, stopping", opts.ε);
    	break;
    }
    
    x_k.device(dev) = x;
    for (Index ir = 0; ir < regs.size(); ir++) {
      if (regs[ir].T) {
        Vector Fx(z[ir].size());
        regs[ir].T->forward(x, Fx, std::sqrt(ρ[ir]));
        zy_update(io, ir, Fx);
      } else {
        zy_update(io, ir, x);
      }
    }
    if (Iterating::ShouldStop("ADMM")) { break; }
  }
  Iterating::Finished();
  return x;
}

} // namespace rl
