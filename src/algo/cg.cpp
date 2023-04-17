#include "cg.hpp"

#include "op/tensorop.hpp"

namespace rl {

auto ConjugateGradients::run(Cx *bdata, Cx *x0data) const -> Vector
{
  Index const rows = op->rows();
  Map const b(bdata, rows);
  Vector q(rows), p(rows), r(rows), x(rows);
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
  float r_old = r.squaredNorm();
  float const thresh = resTol * std::sqrt(r_old);
  Log::Print(FMT_STRING("CG |r| {:5.3E} threshold {:5.3E}"), std::sqrt(r_old), thresh);
  Log::Print(FMT_STRING("IT |r|       α         β         |x|"));
  PushInterrupt();
  for (Index icg = 0; icg < iterLimit; icg++) {
    q = op->forward(p);
    float const α = r_old / CheckedDot(p, q);
    x = x + p * α;
    if (debug) {
        if (auto top = std::dynamic_pointer_cast<TensorOperator<Cx, 5, 4>>(op)) {
            Log::Tensor(fmt::format(FMT_STRING("cg-x-{:02}"), icg), top->ishape, x.data());
            Log::Tensor(fmt::format(FMT_STRING("cg-r-{:02}"), icg), top->ishape, r.data());
        }
    }
    r = r - q * α;
    float const r_new = r.squaredNorm();
    float const β = r_new / r_old;
    p = r + p * β;
    float const nr = sqrt(r_new);
    Log::Print(FMT_STRING("{:02d} {:5.3E} {:5.3E} {:5.3E} {:5.3E}"), icg, nr, α, β, x.norm());
    if (nr < thresh) {
      Log::Print(FMT_STRING("Reached convergence threshold"));
      break;
    }
    r_old = r_new;
    if (InterruptReceived()) {
      break;
    }
  }
  PopInterrupt();
  return x;
}

} // namespace rl
