#include "svd.hpp"

#include "algo/decomp.hpp"
#include "log.hpp"

namespace rl {

SVDBasis::SVDBasis(
  Eigen::Array<Cx, -1, -1> const &dynamics, Index const nRetain, bool const demean, bool const rotate, bool const normalize)
{
  // Calculate SVD - observations are in cols
  Eigen::Array<Cx, -1, -1> d = normalize ? dynamics.colwise().normalized().transpose().eval() : dynamics.transpose().eval();
  if (demean) { d = d.rowwise() - d.colwise().mean(); }
  Index const sz = dynamics.cols();
  basis.resize(nRetain, sz);

  auto const svd = SVD<Cx>(d);
  Log::Print("SVD", "Retaining {} basis vectors, variance {}", nRetain,
             100.f * svd.S.head(nRetain).matrix().stableNorm() / svd.S.matrix().stableNorm());
  basis = rotate ? svd.equalized(nRetain).transpose().eval() : svd.V.leftCols(nRetain).transpose().eval();
  basis *= std::sqrt(sz);
}

} // namespace rl
