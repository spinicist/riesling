#include "pdhg.hpp"

#include "../log/log.hpp"
#include "../prox/stack.hpp"
#include "../tensors.hpp"
#include "common.hpp"
#include "eig.hpp"
#include "iter.hpp"

namespace rl {

PDHG::PDHG(Op::Ptr A_, Op::Ptr P_, std::vector<Regularizer> const &regs, Index imax_, float resTol_, float λA, float λG, Debug debug_)
  : A{A_}
  , P{P_}
  , imax{imax_}
  , resTol{resTol_}
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
    proxʹ = std::make_shared<Proxs::StackProx<Cx>>(ps);
  }

  if (λA == 0.f) { λA = PowerMethodAdjoint(A, P, 32).val; }
  if (λG == 0.f) { λG = PowerMethodAdjoint(G, nullptr, 64).val; }
  σ = 1.f / (λA + λG);
  τ = σ;
  Log::Print("PDHG", "λA {:4.3E} λG {:4.3E} τ {:4.3E} imax {} tol {}", λA, λG, τ, imax, resTol);
}

auto PDHG::run(Vector const &b) const -> Vector { return run(CMap{b.data(), b.rows()}); }

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
  
  float r0 = ParallelNorm(y);
  Iterating::Starting();
  for (Index ii = 0; ii < imax; ii++) {
    xold = x;
    // unext = (I + σP) \ (u + σP(Ax̅ - y))
    A->forward(x̅, utemp);
    utemp.device(Threads::CoreDevice()) = utemp - y;
    if (P) {
      P->iforward(utemp, u, σ);
      P->inverse(u, u, 1.f, σ);
    } else {
      u.device(Threads::CoreDevice()) = (u + σ * utemp) / (1.f + σ);
    }

    // vnext = prox(v + σGx̅);
    G->iforward(x̅, v, σ);
    proxʹ->apply(τ, v, v);
    // xnext = x - τ(A'u + G'v)
    A->iadjoint(u, x);
    G->adjoint(v, x̅);
    fmt::print(stderr, "|x| {} |x̅| {}\n", ParallelNorm(x), ParallelNorm(x̅));
    if (debug) { debug(ii, x, x̅, utemp); }
    x.device(Threads::CoreDevice()) = xold - τ * x - τ * x̅;
    xold.device(Threads::CoreDevice()) = x - xold; // Now it's xdiff
    x̅.device(Threads::CoreDevice()) = x + θ * xold;
    float const r = ParallelNorm(xold) / r0;
    Log::Print("PDHG", "{:02d}: |x| {:4.3E} |r|/|r0| {:4.3E}", ii, ParallelNorm(x), r);
    if (r < resTol) { break; }
    if (Iterating::ShouldStop("PDHG")) { break; }
    // if (debug) { debug(ii, x, x̅, xdiff); }
  }
  Iterating::Finished();
  return x;
}

} // namespace rl
