#include "basis.hpp"

#include "algo/decomp.hpp"

namespace rl {

Basis::Basis(
  Eigen::ArrayXXf const &par,
  Eigen::ArrayXXf const &dyn,
  float const thresh,
  Index const nBasis,
  bool const demean,
  bool const rotate,
  std::vector<Index> const reorder)
  : parameters{par}
  , dynamics{dyn}
{
  // Calculate SVD - observations are in cols
  auto const svd = SVD<float>(demean ? dynamics.colwise() - dynamics.rowwise().mean() : dynamics, true, true);
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

  if (reorder.size()) {
    Index const nReorder = reorder.size();
    if (nReorder < nRetain) {
      Log::Fail("Basis and reordering size did not match");
    }
    Log::Print("Reordering basis");
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm(nReorder);
    for (Index ii = 0; ii < nReorder; ii++) {
      perm.indices()[ii] = reorder[ii];
    }
    basis = (svd.V.leftCols(nReorder) * perm).leftCols(nRetain);
  } else {
    basis = svd.V.leftCols(nRetain);
  }
  
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

  basis *= std::sqrt(basis.rows());

  Log::Print("Computing dictionary");
  dict = basis.transpose() * dynamics.matrix();
  norm = dict.colwise().norm();
  dict = dict.array().rowwise() / norm.transpose();
}

void Basis::write(HD5::Writer &writer)
{
  writer.writeMatrix(Eigen::MatrixXf(basis.transpose()), HD5::Keys::Basis);
  writer.writeMatrix(dict, HD5::Keys::Dictionary);
  writer.writeMatrix(parameters, HD5::Keys::Parameters);
  writer.writeMatrix(norm, HD5::Keys::Norm);
  writer.writeMatrix(dynamics, HD5::Keys::Dynamics);
}

} // namespace rl
