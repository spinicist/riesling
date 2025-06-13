#include "pdhg.hpp"

#include "../log/log.hpp"
#include "../prox/stack.hpp"
#include "../tensors.hpp"
#include "common.hpp"
#include "eig.hpp"

namespace rl {

PDHG::PDHG(std::shared_ptr<Op>             A_,
           std::shared_ptr<Op>             P_,
           std::vector<Regularizer> const &regs,
           Index const                     imax_,
           float const                     σ_,
           float const                     τ_,
           float const                     θ_)
  : A{A_}
  , P{P_}
  , imax{imax_}
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

  if (σ_ > 0) {
    σ = σ_;
  } else {
    σ = 1;
  }

  if (τ_ > 0) {
    τ = τ_;
  } else {
    τ = 1.f / (PowerMethodAdjoint(G, nullptr, 32).val + PowerMethodAdjoint(A, P, 32).val);
  }

  θ = 1;

  Log::Print("PDHG", "σ {:4.3E} τ {:4.3E}", σ, τ);
}

auto PDHG::run(Vector const &b) const -> Vector { return run(CMap{b.data(), b.rows()}); }

auto PDHG::run(CMap y) const -> Vector
{
  if (y.rows() != A->rows()) { throw(Log::Failure("PDHG", "y had {} rows, expected {}", y.rows(), A->rows())); }
  Vector x(A->cols()), x̅(A->cols()), xold(A->cols());
  Vector u(A->rows()), utemp(A->rows()), v(G->rows());

  u.setZero();
  v.setZero();
  x.setZero();
  x̅.setZero();
  xold.setZero();

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
    A->adjoint(u, x);
    G->iadjoint(v, x);
    x.device(Threads::CoreDevice()) = xold - τ * x;
    xold.device(Threads::CoreDevice()) = x - xold; // Now it's xdiff
    x̅.device(Threads::CoreDevice()) = x + xold * θ;
    float const normr = ParallelNorm(xold) / std::sqrt(τ);
    Log::Print("PDHG", "{:02d}: |x| {:4.3E} |r| {:4.3E}", ii, ParallelNorm(x), normr);
    // if (debug) { debug(ii, x, x̅, xdiff); }
  }
  return x;
}

} // namespace rl
