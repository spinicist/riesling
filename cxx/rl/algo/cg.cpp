#include "cg.hpp"

#include "../log/log.hpp"
#include "common.hpp"
#include "iter.hpp"

namespace rl {

auto ConjugateGradients::run(Vector const &b, Vector const &x0) const -> Vector
{
  return run(CMap{b.data(), b.rows()}, CMap{x0.data(), x0.rows()});
}

auto ConjugateGradients::run(CMap b, CMap x0) const -> Vector
{
  Index const rows = A->rows();
  if (rows != A->cols()) {
    throw Log::Failure("CG", "Op had {} rows and {} cols, should be square", rows, A->cols());
  }
  Vector      q(rows), p(rows), r(rows), x(rows);
  // If we have an initial guess, use it
  if (x0.size()) {
    Log::Print("CG", "Warm-start |x| {}", ParallelNorm(x0));
    r = b - A->forward(x0);
    x = x0;
  } else {
    r = b;
    x.setZero();
  }
  p = r;
  float       r_old = ParallelNorm(r);
  float const thresh = opts.resTol * r_old;
  Log::Print("CG", "|r| {:4.3E} threshold {:4.3E}", r_old, thresh);
  Log::Print("CG", "IT |r|       α         β         |x|");
  Iterating::Starting();
  for (Index icg = 0; icg < opts.imax; icg++) {
    A->forward(p, q);
    float const α = CheckedDot(r, r) / CheckedDot(p, q);
    x = x + p * α;
    // if (debug) {
    //   if (auto top = std::dynamic_pointer_cast<TOp<5, 4>>(op)) {
    //     Log::Tensor(fmt::format("cg-x-{:02}", icg), tA->ishape, x.data());
    //     Log::Tensor(fmt::format("cg-r-{:02}", icg), tA->ishape, r.data());
    //   }
    // }
    r = r - q * α;
    float const r_new = CheckedDot(r, r);
    float const β = r_new / r_old;
    p = r + p * β;
    float const nr = std::sqrt(r_new);
    Log::Print("CG", "{:02d} {:4.3E} {:4.3E} {:4.3E} {:4.3E}", icg, nr, α, β, x.stableNorm());
    if (nr < thresh) {
      Log::Print("CG", "Reached convergence threshold");
      break;
    }
    r_old = r_new;
    if (Iterating::ShouldStop("CG")) { break; }
  }
  Iterating::Finished();
  return x;
}

} // namespace rl
