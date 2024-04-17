#include "svd.hpp"

#include "algo/decomp.hpp"
#include "log.hpp"

namespace rl {

SVDBasis::SVDBasis(
  Eigen::ArrayXXf const &dynamics, Index const nRetain, bool const demean, bool const rotate, bool const normalize)
{
  // Calculate SVD - observations are in cols
  Eigen::ArrayXXf d = normalize ? dynamics.colwise().normalized().transpose().eval() : dynamics.transpose().eval();
  if (demean) { d = d.rowwise() - d.colwise().mean(); }
  Index const sz = dynamics.cols();
  basis.resize(nRetain, sz);

  auto const svd = SVD<float>(d);
  Log::Print("dyn {} {} nRetain {} S {}", dynamics.rows(), dynamics.cols(), nRetain, svd.S.rows());
  Log::Print("Retaining {} basis vectors, variance {}", nRetain, svd.S.head(nRetain).square().sum());
  basis = rotate ? svd.equalized(nRetain).transpose().eval() : svd.V.leftCols(nRetain).transpose().eval();
  basis *= std::sqrt(sz);

  Log::Debug("Orthogonality check:\n{}", fmt::streamed(basis * basis.adjoint()));
}

} // namespace rl
