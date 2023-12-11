#include "svd.hpp"

#include "algo/decomp.hpp"
#include "algo/stats.hpp"

namespace rl {

SVDBasis::SVDBasis(Eigen::ArrayXXf const &dynamics,
                          float const            thresh,
                          Index const            nBasis,
                          bool const             demean,
                          bool const             rotate,
                          bool const             normalize)
{
  // Calculate SVD - observations are in cols
  Eigen::ArrayXXf d = normalize ? dynamics.colwise().normalized() : dynamics;
  if (demean) { d = d.colwise() - d.rowwise().mean(); }
  auto const svd = SVD<float>(d.transpose());
  Index      nRetain = 0;
  if (nBasis) {
    nRetain = nBasis;
  } else {
    nRetain = Threshold(svd.S.square(), thresh);
  }
  Log::Print("Retaining {} basis vectors", nRetain);

  basis = rotate ? Eigen::MatrixXf(svd.equalized(nRetain).transpose()) : Eigen::MatrixXf(svd.V.leftCols(nRetain).transpose());
  basis *= std::sqrt(basis.cols());
}

} // namespace rl
