#include "basis.hpp"

#include "algo/decomp.hpp"

namespace rl {

void Basis(
  Eigen::ArrayXXf const &dynamics,
  float const thresh,
  Index const nBasis,
  bool const demean,
  bool const rotate,
  bool const normalize,
  HD5::Writer &writer)
{
  // Calculate SVD - observations are in cols
  Eigen::ArrayXXf d = normalize ? dynamics.colwise().normalized() : dynamics;
  if (demean) {
    d = d.colwise() - d.rowwise().mean();
  }
  auto const svd = SVD<float>(d, true, true);
  Eigen::ArrayXf const vals = svd.vals.square();
  Eigen::ArrayXf cumsum(vals.rows());
  std::partial_sum(vals.begin(), vals.end(), cumsum.begin());
  cumsum = 100.f * cumsum / cumsum.tail(1)[0];
  Index nRetain = 0;
  if (nBasis) {
    nRetain = nBasis;
  } else {
    nRetain = (cumsum < thresh).count();
  }
  Log::Print("Retaining {} basis vectors, cumulative energy: {}", nRetain, cumsum.head(nRetain).transpose());

  Eigen::MatrixXf basis = svd.V.leftCols(nRetain);
  if (rotate) {
    Log::Print("Bastardised Gram-Schmidt");
    basis.col(0) = basis.rowwise().mean().normalized();
    for (Index ii = 1; ii < basis.cols(); ii++) {
      for (Index ij = 0; ij < ii; ij++) {
        basis.col(ii) -= basis.col(ij).dot(basis.col(ii)) * basis.col(ij);
      }
      basis.col(ii).normalize();
    }
  }

  Log::Print("Computing dictionary");
  basis *= std::sqrt(basis.rows());
  Eigen::MatrixXf dict = basis.transpose() * dynamics.matrix();
  Eigen::ArrayXf norm = dict.colwise().norm();
  dict = dict.array().rowwise() / norm.transpose();

  writer.writeMatrix(Eigen::MatrixXf(basis.transpose()), HD5::Keys::Basis);
  writer.writeMatrix(dict, HD5::Keys::Dictionary);
  writer.writeMatrix(norm, HD5::Keys::Norm);
  writer.writeMatrix(dynamics, HD5::Keys::Dynamics);
}

} // namespace rl
