#include "decomp.hpp"
#include "log.hpp"
#include "tensorOps.hpp"

#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

namespace rl {

template <typename S>
Eig<S>::Eig(Eigen::Ref<Matrix const> const &g)
{
  if (g.rows() != g.cols()) { Log::Fail("This is for self-adjoin Eigensystems"); }
  Eigen::SelfAdjointEigenSolver<Matrix> eig(g);
  V = eig.eigenvalues().reverse();
  P = eig.eigenvectors().rowwise().reverse();
}
template struct Eig<float>;
template struct Eig<Cx>;

template <typename S>
SVD<S>::SVD(Eigen::Ref<Matrix const> const &mat)
{
  auto const svd = mat.template bdcSvd<Eigen::ComputeThinU | Eigen::ComputeThinV>();
  S = svd.singularValues();
  U = svd.matrixU();
  V = svd.matrixV();
}
template struct SVD<float>;
template struct SVD<Cx>;

} // namespace rl
