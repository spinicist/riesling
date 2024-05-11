#include "cg.hpp"

#include "op/top.hpp"

namespace rl {

template <typename Scalar>
auto ConjugateGradients<Scalar>::run(Scalar *bdata, Scalar *x0data) const -> Vector
{
  Index const rows = op->rows();
  if (rows != op->cols()) {
    Log::Fail("CG op had {} rows and {} cols, should be square", rows, op->cols());
  }
  Map const   b(bdata, rows);
  Vector      q(rows), p(rows), r(rows), x(rows);
  // If we have an initial guess, use it
  if (x0data) {
    Map const x0(x0data, rows);
    Log::Print("Warm-start CG");
    r = b - op->forward(x0);
    x = x0;
  } else {
    r = b;
    x.setZero();
  }
  p = r;
  float       r_old = r.squaredNorm();
  float const thresh = resTol * std::sqrt(r_old);
  Log::Print("CG |r| {:4.3E} threshold {:4.3E}", std::sqrt(r_old), thresh);
  Log::Print("IT |r|       α         β         |x|");
  PushInterrupt();
  for (Index icg = 0; icg < iterLimit; icg++) {
    op->forward(p, q);
    float const α = r_old / CheckedDot(p, q);
    x = x + p * α;
    if (debug) {
      if (auto top = std::dynamic_pointer_cast<TOp<Cx, 5, 4>>(op)) {
        Log::Tensor(fmt::format("cg-x-{:02}", icg), top->ishape, x.data());
        Log::Tensor(fmt::format("cg-r-{:02}", icg), top->ishape, r.data());
      }
    }
    r = r - q * α;
    float const r_new = r.squaredNorm();
    float const β = r_new / r_old;
    p = r + p * β;
    float const nr = sqrt(r_new);
    Log::Print("{:02d} {:4.3E} {:4.3E} {:4.3E} {:4.3E}", icg, nr, α, β, x.norm());
    if (nr < thresh) {
      Log::Print("Reached convergence threshold");
      break;
    }
    r_old = r_new;
    if (InterruptReceived()) { break; }
  }
  PopInterrupt();
  return x;
}

template struct ConjugateGradients<float>;
template struct ConjugateGradients<Cx>;

} // namespace rl
