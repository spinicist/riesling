#include "pdhg.hpp"

#include "../log/log.hpp"
#include "../prox/norms.hpp"
#include "../prox/stack.hpp"
#include "../tensors.hpp"
#include "common.hpp"
#include "eig.hpp"
#include "iter.hpp"

namespace rl {

PDHG::PDHG(Op::Ptr A, Op::Ptr P_, Proxs::Prox::Ptr pF, std::vector<Regularizer> const &regs, Opts opts, Debug debug_)
  : imax{opts.imax}
  , resTol{opts.resTol}
  , deltaTol{opts.deltaTol}
  , θ{1.f}
  , debug{debug_}
{
  Index const            nR = regs.size();
  std::vector<Op::Ptr>   As(1 + nR);
  std::vector<Prox::Ptr> proxFs(1 + nR);
  As[0] = A;
  proxFs[0] = pF;
  for (Index ir = 0; ir < nR; ir++) {
    As[1 + ir] = regs[ir].T ? regs[ir].T : Ops::Identity::Make(A->cols());
    proxFs[1 + ir] = regs[ir].P;
  }
  K = Ops::VStack::Make(As);
  proxF = Proxs::Stack::Make(proxFs);

  σ = 1.f / opts.λA;
  τ = 1.f / opts.λG;
  Log::Print("PDHG", "K {} {} σ {:4.3E} τ {:4.3E} Res tol {} Δx tol {}", K->rows(), K->cols(), σ, τ, resTol, deltaTol);
}

auto PDHG::run() const -> Vector
{
  Vector x(K->cols()), x̅(K->cols()), xold(K->cols());
  x.setZero();
  x̅.setZero();
  xold.setZero();

  Vector y(K->rows()), ytemp(K->rows());
  y.setZero();
  ytemp.setZero();

  // float const r0 = ParallelNorm(y);
  Iterating::Starting();
  for (Index ii = 0; ii < imax; ii++) {
    xold = x;
    // yn = prox_F*σ(y + σKx̅)
    K->forward(x̅, ytemp);
    // if (P)
    ytemp.device(Threads::CoreDevice()) = y + σ * ytemp;
    proxF->conj(σ, ytemp, y);
    K->iadjoint(y, x, -τ);
    x̅.device(Threads::CoreDevice()) = 2.f * x - xold;
    xold.device(Threads::CoreDevice()) = x - xold; // Now it's xdiff
    float const nx = ParallelNorm(x);
    float const ndx = ParallelNorm(xold);
    Log::Print("PDHG", "{:02d}: |x| {:4.3E} |Δx/x| {:4.3E}", ii, nx, ndx / nx);
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
