#include "pdhg.hpp"

#include "../log/log.hpp"
#include "../prox/norms.hpp"
#include "../prox/stack.hpp"
#include "../tensors.hpp"
#include "common.hpp"
#include "eig.hpp"
#include "iter.hpp"

namespace rl::PDHG {

auto Run(Vector const &b, Op::Ptr A, Op::Ptr P, std::vector<Regularizer> const &regs, Opts opts, Debug d) -> Vector
{
  return Run(CMap{b.data(), b.rows()}, A, P, regs, opts, d);
}

/*  This follows the least-squares specific form of PDHG from Ong et al 2020
 *  Note that the code in SigPy is different to this. It was generalised to other problems, not only least-squares
 *  The specific trick to making the below work is realizing that the conjugate proximal operators are needed because
 *  those updates are on the dual variables
 */
auto Run(CMap b, Op::Ptr A, Op::Ptr P, std::vector<Regularizer> const &regs, Opts opts, Debug debug) -> Vector
{
  if (b.rows() != A->rows()) { throw(Log::Failure("PDHG", "b had {} rows, expected {}", b.rows(), A->rows())); }
  if (regs.size() < 1) { throw(Log::Failure("PDHG", "Requires at least one regularizer")); }
  Index const                   nx = regs.size() + 1;
  std::vector<Op::Ptr>          As(nx), Ps(nx);
  std::vector<Proxs::Prox::Ptr> proxs(nx);

  /* Data consistency term */
  As[0] = A;
  Ps[0] = P;
  proxs[0] = opts.lad ? Proxs::L1::Make(1.f, b, P) : Proxs::SumOfSquares::Make(b, P);
  /* Regularizers */
  for (Index ir = 0; ir < regs.size(); ir++) {
    As[ir + 1] = regs[ir].T;
    Ps[ir + 1] = nullptr; /* Regularizers are not preconditioned. Yet. */
    proxs[ir + 1] = regs[ir].P;
  }

  float const L = std::sqrt(opts.λA * opts.λA + regs.size()); /* Assumes all regularizer transforms have max eval=1 */
  float const σ = 1.f / L;
  float const τ = 1.f / L;
  Log::Print("PDHG", "{}σ {:4.3E} τ {:4.3E} Δx tol {}", opts.lad ? "LAD " : "", σ, τ, opts.deltaTol);

  Vector x(A->cols()), x̅(A->cols()), xold(A->cols());
  x.setZero();
  x̅.setZero();
  xold.setZero();

  std::vector<Vector> ys(nx);
  for (Index ir = 0; ir < nx; ir++) {
    ys[ir] = Vector(As[ir]->rows());
    ys[ir].setZero();
  }

  float const r0 = ParallelNorm(b);

  Iterating::Starting();
  for (Index ii = 0; ii < opts.imax; ii++) {
    xold = x;

    /* PDHG steps are
     * yn = proxF*σ(y + σAx̅)
     * xn = proxGτ(x - τA'yn)
     * x̅ = xn + (xn - x)
     *
     *      But G(x) = 0 so proxGτ(x) = x
     */
    for (Index ix = 0; ix < nx; ix++) {
      if (As[ix] && Ps[ix]) {
        auto ytemp = As[ix]->forward(x̅);
        Ps[ix]->iforward(ytemp, ys[ix], σ);
      } else if (As[ix]) {
        As[ix]->iforward(x̅, ys[ix], σ);
      } else {
        ys[ix].device(Threads::CoreDevice()) += σ * x̅;
      }
      proxs[ix]->conj(σ, ys[ix], ys[ix]); /* DANGER Be careful with patch-based regs like LLR */

      if (As[ix]) { As[ix]->iadjoint(ys[ix], x, -τ); }
    }
    x̅.device(Threads::CoreDevice()) = 2.f * x - xold;
    if (debug) { debug(ii, x, x̅); }
    xold.device(Threads::CoreDevice()) = x - xold; // Now it's xdiff
    float const normx = ParallelNorm(x);
    float const normdx = ParallelNorm(xold);
    Log::Print("PDHG", "{:02d}: |x| {:4.3E} |Δx/x| {:4.3E}", ii, normx, normdx / normx);
    if (normdx / normx < opts.deltaTol) {
      Log::Print("PDHG", "Δx tolerance reached");
      break;
    }
    if (Iterating::ShouldStop("PDHG")) { break; }
    // if (debug) { debug(ii, x, x̅, xdiff); }
  }
  Iterating::Finished();
  return x;
}

} // namespace rl::PDHG
