#include "gs.hpp"

namespace rl {

auto GramSchmidt(Eigen::MatrixXcf const &V, bool const renorm) -> Eigen::MatrixXcf
{
  Index const      M = V.rows();
  Index const      N = V.cols();
  Eigen::MatrixXcf U = Eigen::MatrixXcf::Zero(M, N);
  U.col(0) = V.col(0).normalized();
  for (Index ii = 1; ii < N; ii++) {
    U.col(ii) = V.col(ii);
    for (Index ij = 0; ij < ii; ij++) {
      U.col(ii) = U.col(ii) - (U.col(ij).dot(U.col(ii))) * U.col(ij);
    }
    U.col(ii).normalize();
  }
  if (renorm) { U *= std::sqrt(U.rows()); }
  return U;
}

} // namespace rl