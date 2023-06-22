#include "pdhg.hpp"

#include "common.hpp"
#include "log.hpp"
#include "tensorOps.hpp"

namespace rl {

auto PDHG::run(Cx const *bdata, float σ) const -> Vector
{
  CMap const b(bdata, A->rows());
  Vector u(A->rows());
  u.setZero();
  Vector x(A->cols()), xbar(A->cols()), xold(A->cols()), xdiff(A->cols()), v(G->rows());
  x.setZero();
  xbar.setZero();
  xold.setZero();
  xdiff.setZero();
  v.setZero();

  auto lhs = (*P + 1.f)->inverse();

  Log::Print("PDHG σ {}", σ);
  for (Index ii = 0; ii < iterLimit; ii++) {
    xold = x;
    u = lhs->forward(u + P->forward(A->forward(xbar) - b));
    v = prox->apply(1.f, v + σ * G->forward(xbar));
    x = x - (A->adjoint(u) + G->adjoint(v));
    xdiff = x - xold;
    xbar = x + xdiff;
    float const normr = xdiff.norm() / std::sqrt(1.f);
    Log::Print("PDHG {:02d}: |r| {}", ii, normr);
  }
  return x;
}

} // namespace rl
