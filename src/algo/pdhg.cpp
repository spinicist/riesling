#include "pdhg.hpp"

#include "algo/eig.hpp"
#include "common.hpp"
#include "log.hpp"
#include "prox/l2.hpp"
#include "prox/stack.hpp"
#include "tensorOps.hpp"

namespace rl {

auto PDHG::run(Cx const *bdata, float τ) const -> Vector
{
  Index const nR = G.size();
  assert(prox.size() == nR);
  assert(σG.size() == nR);
  CMap const b(bdata, A->rows());
  std::shared_ptr<Op> Aʹ = std::make_shared<Ops::VStack<Cx>>(A, G);
  std::shared_ptr<Prox> l2 = std::make_shared<Proxs::L2>(1.f, b);
  std::vector<std::shared_ptr<Prox>> ps;
  ps.push_back(l2);
  for (Index ii = 0; ii < nR; ii++) {
    ps.push_back(std::make_shared<Proxs::ConjugateProx<Cx>>(prox[ii]));
  }
  Proxs::StackProx proxʹ(ps);

  std::vector<std::shared_ptr<Op>> sG;
  sG.push_back(P);
  for (Index ii = 0; ii < nR; ii++) {
    sG.push_back(std::make_shared<Ops::DiagScale<Cx>>(G[ii]->rows(), σG[ii]));
  }
  std::shared_ptr<Ops::Op<Cx>> σ = std::make_shared<Ops::DStack<Cx>>(sG);

  if (τ < 0.f) {
    auto eig = PowerMethodForward(Aʹ, σ, 32);
    τ = 1.f / eig.val;
  }

  Vector u(Aʹ->rows()), v(Aʹ->rows());
  u.setZero();
  Vector x(Aʹ->cols()), x̅(Aʹ->cols()), xold(Aʹ->cols()), xdiff(Aʹ->cols());
  x.setZero();
  x̅.setZero();
  xold.setZero();
  xdiff.setZero();

  Log::Print("PDHG τ {} σ {}", τ, fmt::join(σG, ","));
  for (Index ii = 0; ii < iterLimit; ii++) {
    xold = x;
    Log::Print<Log::Level::High>("PDHG updating dual");
    v = u + σ->forward(Aʹ->forward(x̅));
    proxʹ.apply(σ, v, u);
    Log::Print<Log::Level::High>("PDHG updating primal");
    x = x - τ * Aʹ->adjoint(u);
    xdiff = x - xold;
    x̅ = x + xdiff;
    float const normr = xdiff.norm() / std::sqrt(τ);
    Log::Print("PDHG {:02d}: |r| {}", ii, normr);
    if (debug) { debug(ii, x, x̅, xdiff); }
  }
  return x;
}

} // namespace rl
