#include "pdhg.hpp"

#include "../log/log.hpp"
#include "../prox/stack.hpp"
#include "../tensors.hpp"
#include "common.hpp"
#include "eig.hpp"
#include "iter.hpp"

namespace rl {

PDHG::PDHG(Op::Ptr A_, Op::Ptr P_, std::vector<Regularizer> const &regs, Opts opts, Debug debug_)
  : A{A_}
  , P{P_}
  , imax{opts.imax}
  , resTol{opts.resTol}
  , deltaTol{opts.deltaTol}
  , θ{1.f}
  , debug{debug_}
{
  Index const nR = regs.size();
  if (nR == 1) {
    proxʹ = regs.front().P;
    G = regs.front().T ? regs.front().T : Ops::Identity<Cx>::Make(A->cols());
  } else {
    std::vector<Op::Ptr>   Gs(nR);
    std::vector<Prox::Ptr> ps(nR);
    for (Index ir = 0; ir < nR; ir++) {
      Gs[ir] = regs[ir].T ? regs[ir].T : Ops::Identity<Cx>::Make(A->cols());
      ps[ir] = regs[ir].P;
    }
    G = std::make_shared<Ops::VStack<Cx>>(Gs);
    proxʹ = std::make_shared<Proxs::Stack<Cx>>(ps);
  }

  σ = 1.f / opts.λA;
  τ = 1.f / opts.λG;
  Log::Print("PDHG", "σ {:4.3E} τ {:4.3E} Res tol {} Δx tol {}", σ, τ, resTol, deltaTol);
}

auto PDHG::run(Vector const &b) const -> Vector { return run(CMap{b.data(), b.rows()}); }

/*  This follows the least-squares specific form of PDHG from Ong et al 2020
 *  Note that the code in SigPy is different to this. It was generalised to other problems, not only least-squares
 *  The specific trick to making the below work is realizing that the dual proximal operator is needed, not the primal
 */
auto PDHG::run(CMap y) const -> Vector
{
  if (y.rows() != A->rows()) { throw(Log::Failure("PDHG", "y had {} rows, expected {}", y.rows(), A->rows())); }
  Vector x(A->cols()), x̅(A->cols()), xold(A->cols());
  x.setZero();
  x̅.setZero();
  xold.setZero();

  Vector u(A->rows()), utemp(A->rows()), v(G->rows());
  u.setZero();
  utemp.setZero();
  v.setZero();

  float const r0 = ParallelNorm(y);
  Iterating::Starting();
  for (Index ii = 0; ii < imax; ii++) {
    xold = x;
    // unext = (I + σP) \ (u + σP(Ax̅ - y))
    utemp.device(Threads::CoreDevice()) = A->forward(x̅) - y;
    float const r = ParallelNorm(utemp);
    if (r / r0 < resTol) {
      Log::Print("PDHG", "Residual tolerance reached");
      break;
    }

    if (P) {
      P->iforward(utemp, u, σ);
      P->inverse(u, u, σ, 1.f);
    } else {
      u.device(Threads::CoreDevice()) = (u + σ * utemp) / (1.f + σ);
    }

    // vnext = prox(v + σGx̅);
    G->iforward(x̅, v, σ);
    proxʹ->dual(1.f, v, v);
    // xnext = x - τ(A'u + G'v)
    A->adjoint(u, x); // Re-use variables to save memory
    G->adjoint(v, x̅); // Re-use variables to save memory
    x.device(Threads::CoreDevice()) = xold - τ * (x + x̅);
    x̅.device(Threads::CoreDevice()) = x * 2.f - xold;
    if (debug) { debug(ii, x, x̅, u); }
    xold.device(Threads::CoreDevice()) = x - xold; // Now it's xdiff
    float const nx = ParallelNorm(x);
    float const ndx = ParallelNorm(xold);
    Log::Print("PDHG", "{:02d}: |x| {:4.3E} |Δx/x| {:4.3E} |r|/|r0| {:4.3E}", ii, nx, ndx / nx, r / r0);
    if (ndx / nx < deltaTol) {
      Log::Print("PDHG", "Δx tolerance reached");
      break;
    }
    if (Iterating::ShouldStop("PDHG")) { break; }
    // if (debug) { debug(ii, x, x̅, xdiff); }
  }
  Iterating::Finished();
  return x;
}

} // namespace rl
