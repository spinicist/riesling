#include "pdhg.hpp"

#include "../log/log.hpp"
#include "../prox/norms.hpp"
#include "../prox/stack.hpp"
#include "../tensors.hpp"
#include "common.hpp"
#include "eig.hpp"
#include "iter.hpp"

namespace rl::PDHG {

struct PDResType
{
  float primal;
  float dual;
};
auto PDResiduals(Vector const               &x,
                 Vector const               &xold,
                 std::vector<Vector> const  &ys,
                 std::vector<Vector> const  &yolds,
                 std::vector<Op::Ptr> const &As,
                 std::vector<Op::Ptr> const &Ps,
                 float const                 τ,
                 float const                 σ) -> PDResType
{
  Vector xd(x.size()), p(x.size());
  xd.device(Threads::CoreDevice()) = xold - x;
  float pres = 0.f, dres = 0.f;
  for (Index ii = 0; ii < ys.size(); ii++) {
    p.device(Threads::CoreDevice()) = xd / τ;
    Vector yd(ys[ii].size()), d(ys[ii].size());
    yd.device(Threads::CoreDevice()) = yolds[ii] - ys[ii];
    if (As[ii] && Ps[ii]) { Ps[ii]->forward(yd, yd); }
    d.device(Threads::CoreDevice()) = yd / σ;
    if (As[ii] && Ps[ii]) {
      As[ii]->iadjoint(yd, p, -1.f);
      As[ii]->forward(xd, yd);
      Ps[ii]->iforward(yd, d, -1.f);
    } else if (As[ii]) {
      As[ii]->iadjoint(yd, p, -1.f);
      As[ii]->iforward(xd, d, -1.f);
    } else {
      p.device(Threads::CoreDevice()) -= yd;
      d.device(Threads::CoreDevice()) -= xd;
    }
    pres += std::pow(ParallelNorm(p), 2);
    dres += std::pow(ParallelNorm(d), 2);
  }
  pres = std::sqrt(pres);
  dres = std::sqrt(dres);
  return {pres, dres};
}

auto Run(Vector const &b, Op::Ptr A, Op::Ptr P, std::vector<Regularizer> const &regs, Opts opts, Debug d) -> Vector
{
  return Run(CMap{b.data(), b.rows()}, A, P, regs, opts, d);
}

auto Run(CMap b, Op::Ptr E, Op::Ptr P, std::vector<Regularizer> const &regs, Opts opts, Debug debug) -> Vector
{
  if (b.rows() != E->rows()) { throw(Log::Failure("PDHG", "b had {} rows, expected {}", b.rows(), E->rows())); }
  if (regs.size() < 1) { throw(Log::Failure("PDHG", "Requires at least one regularizer")); }
  Index const                   nx = regs.size() + 1;
  std::vector<Op::Ptr>          As(nx), Ps(nx); /* Transforms, preconditioners */
  std::vector<Proxs::Prox::Ptr> proxs(nx);      /* Proximal operators */

  /* Data consistency term */
  As[0] = E;
  Ps[0] = P;
  proxs[0] = opts.lad ? Proxs::L1::Make(1.f, b, P) : Proxs::SumOfSquares::Make(b, P);
  /* Regularizers */
  for (Index ir = 0; ir < regs.size(); ir++) {
    As[ir + 1] = regs[ir].T;
    Ps[ir + 1] = nullptr; /* Regularizers are not preconditioned. Yet. */
    proxs[ir + 1] = regs[ir].P;
  }

  /* Assumes all regularizer transforms have max eval=1 */
  /* Note difference between eigenvalues in Ong and singular values in Sidky */
  float const L = opts.λE + regs.size();
  float       σ = 1.f / std::sqrt(L);
  float       τ = 1.f / std::sqrt(L);
  float       α = 0.5f;
  float const s = 1.f;
  float const η = 0.95f;
  float const Δ = 1.5f;
  Log::Print("PDHG", "{}{}σ {:4.3E} τ {:4.3E} Convergence Tolerance {}", opts.adaptive ? "Adaptive " : "",
             opts.lad ? "LAD " : "", σ, τ, opts.tol);
  Vector x(E->cols()), x̅(E->cols()), xold(E->cols());
  x.setZero();
  x̅.setZero();
  xold.setZero();

  std::vector<Vector> ys(nx), yolds(nx);
  for (Index ir = 0; ir < nx; ir++) {
    if (As[ir]) {
      ys[ir] = Vector(As[ir]->rows());
      yolds[ir] = Vector(As[ir]->rows());
    } else {
      ys[ir] = Vector(E->cols());
      yolds[ir] = Vector(E->cols());
    }
    ys[ir].setZero();
    yolds[ir].setZero();
  }

  Iterating::Starting();
  for (Index ii = 0; ii < opts.imax; ii++) {
    /* PDHG steps are
     * yn = proxF*σ(y + σAx̅)
     * xn = proxGτ(x - τA'yn)
     * x̅ = xn + (xn - x)
     *
     *      But G(x) = 0 so proxGτ(x) = x
     */
    for (Index ix = 0; ix < nx; ix++) {
      yolds[ix] = ys[ix];
      if (As[ix] && Ps[ix]) {
        auto ytemp = As[ix]->forward(x̅);
        Ps[ix]->iforward(ytemp, ys[ix], σ);
      } else if (As[ix]) {
        As[ix]->iforward(x̅, ys[ix], σ);
      } else {
        ys[ix].device(Threads::CoreDevice()) += σ * x̅;
      }
      proxs[ix]->conj(σ, ys[ix], ys[ix]); /* DANGER Be careful with patch-based regs like LLR */
    }

    xold.device(Threads::CoreDevice()) = x;
    for (Index ix = 0; ix < nx; ix++) {
      if (As[ix]) {
        As[ix]->iadjoint(ys[ix], x, -τ);
      } else {
        x.device(Threads::CoreDevice()) -= τ * ys[ix];
      }
    }
    x̅.device(Threads::CoreDevice()) = 2.f * x - xold;

    if (debug) { debug(ii, x); }
    float const normx = ParallelNorm(x);

    if (opts.adaptive) {
      auto const res = PDResiduals(x, xold, ys, yolds, As, Ps, τ, σ);

      if (res.primal > s * res.dual * Δ) { /* Large primal */
        τ = τ / (1.f - α);
        σ = σ * (1.f - α);
        α = α * η;
      } else if (res.primal < s * res.dual / Δ) { /* Large dual */
        τ = τ * (1.f - α);
        σ = σ / (1.f - α);
        α = α * η;
      }

      Log::Print("PDHG", "{:02d}: |x| {:4.3E} |P| {:4.3E} |D| {:4.3E} σ {:4.3E} τ {:4.3E} α {:4.3E}", ii, normx, res.primal,
                 res.dual, σ, τ, α);
      if (res.primal / normx < opts.tol) {
        Log::Print("PDHG", "Primal tolerance reached");
        break;
      }
    } else {
      xold.device(Threads::CoreDevice()) -= x;
      float const normdx = ParallelNorm(xold);
      Log::Print("PDHG", "{:02d}: |x| {:4.3E} |Δx| {:4.3E}", ii, normx, normdx);
      if (normdx / normx < opts.tol) {
        Log::Print("PDHG", "Δ tolerance reached");
        break;
      }
    }

    if (Iterating::ShouldStop("PDHG")) { break; }
  }
  Iterating::Finished();
  return x;
}

} // namespace rl::PDHG
