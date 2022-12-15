#include "basis.hpp"

#include "algo/decomp.hpp"

namespace rl {

Basis::Basis(
  Eigen::ArrayXXf const &par,
  Eigen::ArrayXXf const &dyn,
  float const thresh,
  Index const nBasis,
  bool const demean,
  bool const varimax)
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
  // Scale and flip the basis vectors to always have a positive first element for stability
  // Eigen::ArrayXf flip = Eigen::ArrayXf::Ones(nRetain);
  // flip = (svd.V.leftCols(nRetain).row(0).transpose().array() < 0.f).select(-flip, flip);
  basis = svd.V.leftCols(nRetain).array();
  if (varimax) {
    Log::Print("SIM Applying varimax rotation");
    float gamma = 1.0f;
    float const tol = 1e-6f;
    float q = 20;
    Index const p = basis.rows();
    Index const k = basis.cols();
    Eigen::MatrixXf R = Eigen::MatrixXf::Identity(k, k);
    float d = 0.f;
    for (Index ii = 0; ii < q; ii++) {
      float const d_old = d;
      Eigen::MatrixXf const λ = basis * R;
      Eigen::MatrixXf const x = basis.transpose() * (λ.array().pow(3.f).matrix() -
                                                     (λ * (λ.transpose() * λ).diagonal().asDiagonal()) * (gamma / p));
      auto const svdv = SVD<float>(x);
      R = svdv.U * svdv.V.adjoint();
      d = svdv.vals.sum();
      if (d_old != 0.f && (d / d_old) < 1 + tol)
        break;
    }
    basis = basis * R;
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
