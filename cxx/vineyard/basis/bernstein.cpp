#include "bernstein.hpp"

// From Knuth, surprised this isn't in STL
auto Choose(Index n, Index k) -> Index
{
  if (k > n) return 0;

  Index r = 1;
  for (Index d = 1; d <= k; ++d) {
    r *= n--;
    r /= d;
  }
  return r;
}

namespace rl {

auto BernsteinPolynomial(Index const N, Index const traces) -> Basis<Cx>
{
  Eigen::ArrayXf const x = Eigen::ArrayXf::LinSpaced(traces, 0.f, 1.f);

  Basis<Cx> basis(N + 2, 1, traces);
  basis.setConstant(1.f);
  for (Index ii = 0; ii <= N; ii++) {
    Eigen::ArrayXf const b = Choose(N, ii) * x.pow(ii) * (1.f - x).pow(N - ii);
    for (Index it = 0; it < traces; it++) {
      basis(ii + 1, 0, it) = b(it);
    }
  }
  return basis;
}

} // namespace rl