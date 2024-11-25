#include "lad.hpp"

#include "../log.hpp"
#include "../op/top.hpp"
#include "../prox/norms.hpp"
#include "../tensors.hpp"
#include "common.hpp"
#include "iter.hpp"
#include "lsmr.hpp"

namespace rl {

auto LAD::run(Cx const *bdata, float ρ) const -> Vector
{
  // Allocate all memory
  Vector      x(A->cols());
  Index const rows = A->rows();
  Vector      u(rows), z(rows), zprev(rows), Ax_sub_b(rows), bzu(rows);
  CMap const  b(bdata, A->rows());

  LSMR inner{A, M, nullptr, iters0, aTol, bTol, cTol};

  Proxs::L1 S(1.f, A->rows());

  // Set initial values
  x.setZero();
  z.setZero();
  u.setZero();

  Log::Print("LAD", "Abs ε {}", ε);
  Iterating::Starting();
  for (Index ii = 0; ii < outerLimit; ii++) {
    bzu = b + z - u;
    x = inner.run(bzu, 0.f, x);
    inner.iterLimit = iters1;
    Ax_sub_b = A->forward(x) - b;
    zprev = z;
    S.apply(1.f / ρ, Ax_sub_b + u, z);
    u = u + Ax_sub_b - z;

    float const normx = ParallelNorm(x);
    float const normz = ParallelNorm(z);
    float const normu = ParallelNorm(u);

    float const pRes = ParallelNorm(Ax_sub_b - z) / std::max(normx, normz);
    float const dRes = ParallelNorm(z - zprev) / normu;

    Log::Print("LAD", "{:02d} |x| {:4.3E} |z| {:4.3E} |u| {:4.3E} ρ {:4.3E} |Primal| {:4.3E} |Dual| {:4.3E}", ii, normx, normz,
               normu, ρ, pRes, dRes);

    if ((pRes < ε) && (dRes < ε)) {
      Log::Print("LAD", "Primal and dual tolerances achieved, stopping");
      break;
    }
    if (ii > 0) {
      // ADMM Penalty Parameter Selection by Residual Balancing, Wohlberg 2017
      float const ratio = std::sqrt(pRes / dRes);
      float const τ = (ratio < 1.f) ? std::max(1.f / τmax, 1.f / ratio) : std::min(τmax, ratio);
      if (pRes > μ * dRes) {
        ρ *= τ;
        u /= τ;
      } else if (dRes > μ * pRes) {
        ρ /= τ;
        u *= τ;
      }
    }
    if (Iterating::ShouldStop("LAD")) { break; }
  }
  Iterating::Finished();
  return x;
}

} // namespace rl
