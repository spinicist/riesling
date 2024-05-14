#include "pdhg.hpp"

#include "algo/eig.hpp"
#include "common.hpp"
#include "log.hpp"
#include "prox/lsq.hpp"
#include "prox/stack.hpp"
#include "tensors.hpp"

namespace rl {

PDHG::PDHG(std::shared_ptr<Op>       A,
           std::shared_ptr<Op>       P,
           Regularizers const       &reg,
           std::vector<float> const &σin,
           float const               τin,
           Callback const           &cb)
{
  Index const nR = reg.count();

  Aʹ = std::make_shared<Ops::VStack<Cx>>(A, reg.ops);
  l2 = std::make_shared<Proxs::LeastSquares<Cx>>(1.f, A->rows());
  std::vector<std::shared_ptr<Prox>> ps;
  ps.push_back(l2);
  for (Index ii = 0; ii < nR; ii++) {
    ps.push_back(std::make_shared<Proxs::ConjugateProx<Cx>>(reg.prox[ii]));
  }
  proxʹ = std::make_shared<Proxs::StackProx<Cx>>(ps);

  σ = reg.σ(σin);
  std::vector<std::shared_ptr<Op>> sG;
  sG.push_back(P);
  for (Index ii = 0; ii < nR; ii++) {
    sG.push_back(std::make_shared<Ops::DiagScale<Cx>>(reg.ops[ii]->rows(), σ[ii]));
  }
  σOp = std::make_shared<Ops::DStack<Cx>>(sG);

  if (τin < 0.f) {
    auto eig = PowerMethodForward(Aʹ, σOp, 32);
    τ = 1.f / eig.val;
  } else {
    τ = τin;
  }

  u.resize(Aʹ->rows());
  u.setZero();
  v.resize(Aʹ->rows());
  v.setZero();

  x.resize(Aʹ->cols());
  x.setZero();
  x̅.resize(Aʹ->cols());
  x̅.setZero();
  xold.resize(Aʹ->cols());
  xold.setZero();
  xdiff.resize(Aʹ->cols());
  xdiff.setZero();

  debug = cb;

  Log::Print("PDHG σ {:4.3E} τ {:4.3E}", fmt::join(σ, ","), τ);
}

auto PDHG::run(Cx const *bdata, Index const iterLimit) -> Vector
{
  l2->setBias(bdata);
  for (Index ii = 0; ii < iterLimit; ii++) {
    xold = x;
    v = u + σOp->forward(Aʹ->forward(x̅));
    proxʹ->apply(σOp, v, u);
    x = x - τ * Aʹ->adjoint(u);
    xdiff = x - xold;
    x̅ = x + xdiff;
    float const normr = xdiff.stableNorm() / std::sqrt(τ);
    Log::Print("PDHG {:02d}: |x| {:4.3E} |r| {:4.3E}", ii, x.stableNorm(), normr);
    if (debug) { debug(ii, x, x̅, xdiff); }
  }
  return x;
}

} // namespace rl
