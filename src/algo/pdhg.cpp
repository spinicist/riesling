#include "pdhg.hpp"

#include "algo/eig.hpp"
#include "common.hpp"
#include "log.hpp"
#include "prox/l2.hpp"
#include "prox/stack.hpp"
#include "tensorOps.hpp"

namespace rl {

auto PDHG::run(Cx const *bdata, float eigG, float τ) const -> Vector
{
  CMap const b(bdata, A->rows());

  std::shared_ptr<Op> Aʹ = std::make_shared<Ops::VStack<Cx>>(A, G);
  std::vector<std::shared_ptr<Prox::Prox<Cx>>> ps;
  ps.push_back(std::make_shared<Prox::L2>(1.f, b));
  ps.push_back(std::make_shared<Prox::ConjugateProx<Cx>>(prox));
  Prox::StackProx proxS(ps);

  auto σG = std::make_shared<Ops::DiagScale<Cx>>(G->rows(), eigG);
  std::shared_ptr<Ops::Op<Cx>> σ = std::make_shared<Ops::DStack<Cx>>(P, σG);

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

  Log::Print("PDHG τ {}", τ);
  for (Index ii = 0; ii < iterLimit; ii++) {
    xold = x;
    v = u + σ->forward(Aʹ->forward(x̅));
    proxS.apply(σ, v, u);
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
