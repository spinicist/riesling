#include "pdhg.hpp"

#include "../log/log.hpp"
#include "../prox/lsq.hpp"
#include "../prox/stack.hpp"
#include "../tensors.hpp"
#include "common.hpp"
#include "eig.hpp"

namespace rl {

PDHG::PDHG(std::shared_ptr<Op>             A,
           std::shared_ptr<Op>             P,
           std::vector<Regularizer> const &regs,
           std::vector<float> const       &σin,
           float const                     τin,
           Callback const                 &cb)
{
  Index const            nR = regs.size();
  std::vector<Op::Ptr>   ops(nR);
  std::vector<Prox::Ptr> ps;
  l2 = std::make_shared<Proxs::LeastSquares<Cx>>(1.f, A->rows());
  ps.push_back(l2);
  for (Index ir = 0; ir < nR; ir++) {
    ops[ir] = regs[ir].T ? regs[ir].T : Ops::Identity<Cx>::Make(A->cols());
    ps.push_back(std::make_shared<Proxs::ConjugateProx<Cx>>(regs[ir].P));
  }
  Aʹ = std::make_shared<Ops::VStack<Cx>>(A, ops);
  proxʹ = std::make_shared<Proxs::StackProx<Cx>>(ps);

  if (σin.size() == regs.size()) {
    σ = σin;
  } else {
    σ.clear();
    for (auto &G : ops) {
      auto eigG = PowerMethod(G, 4);
      σ.push_back(1.f / eigG.val);
    }
  }

  std::vector<std::shared_ptr<Op>> sG;
  if (P) {
    sG.push_back(P);
  } else {
    sG.push_back(Ops::Identity<Cx>::Make(A->rows()));
  }
  for (Index ir = 0; ir < nR; ir++) {
    Index const nrows = regs[ir].T ? regs[ir].T->rows() : A->rows();
    sG.push_back(std::make_shared<Ops::DiagScale<Cx>>(nrows, σ[ir]));
  }
  σOp = std::make_shared<Ops::DStack<Cx>>(sG);

  if (τin < 0.f) {
    auto eig = PowerMethodForward(Aʹ, σOp, 32);
    τ = 1.f / eig.val;
  } else {
    τ = τin;
  }

  debug = cb;

  Log::Print("PDHG", "σ {:4.3E} τ {:4.3E}", fmt::join(σ, ","), τ);
}

auto PDHG::run(Vector const &b) const -> Vector { return run(CMap{b.data(), b.rows()}); }

auto PDHG::run(CMap b) const -> Vector
{
  l2->setY(b);

  Vector x(Aʹ->cols()), x̅(Aʹ->cols()), xold(Aʹ->cols()), xdiff(Aʹ->cols()), u(Aʹ->rows()), v(Aʹ->rows());

  u.setZero();
  v.setZero();
  x.setZero();
  x̅.setZero();
  xold.setZero();
  xdiff.setZero();

  for (Index ii = 0; ii < imax; ii++) {
    xold = x;
    v = u + σOp->forward(Aʹ->forward(x̅));
    proxʹ->apply(σOp, v, u);
    x = x - τ * Aʹ->adjoint(u);
    xdiff = x - xold;
    x̅ = x + xdiff;
    float const normr = ParallelNorm(xdiff) / std::sqrt(τ);
    Log::Print("PDHG", "{:02d}: |x| {:4.3E} |r| {:4.3E}", ii, ParallelNorm(x), normr);
    if (debug) { debug(ii, x, x̅, xdiff); }
  }
  return x;
}

} // namespace rl
