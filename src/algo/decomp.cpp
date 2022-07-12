#include "decomp.h"
#include "log.h"
#include "tensorOps.h"

#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

namespace rl {

PCAResult PCA(Eigen::Map<Eigen::MatrixXcf const> const &data, Index const nR, float const thresh)
{
  auto const dm = data.colwise() - data.rowwise().mean();
  Eigen::MatrixXcf gramian = (dm.conjugate() * dm.transpose()) / (dm.rows() - 1);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcf> eig(gramian);
  Eigen::ArrayXf const vals = eig.eigenvalues().reverse().array().abs();
  Eigen::ArrayXf cumsum(vals.rows());
  std::partial_sum(vals.begin(), vals.end(), cumsum.begin());
  cumsum /= cumsum.tail(1)(0);
  Index nRetain = vals.rows();
  if ((thresh > 0.f) && (thresh <= 1.f)) {
    nRetain = (cumsum < thresh).count();
  } else {
    nRetain = std::min(nR, vals.rows());
  }
  return {eig.eigenvectors().rightCols(nRetain).rowwise().reverse(), vals.head(nRetain)};
}

template <typename Scalar>
SVD<Scalar>::SVD(Eigen::Ref<Matrix const> const &mat, bool const transpose, bool const verbose)
{
  if (transpose) {
    if (verbose) {
      Log::Print(FMT_STRING("SVD Transpose Size {}x{}"), mat.cols(), mat.rows());
    }
    auto const svd = mat.transpose().bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
    this->vals = svd.singularValues();
    this->U = svd.matrixU();
    this->V = svd.matrixV();
  } else {
    if (verbose) {
      Log::Print(FMT_STRING("SVD Size {}x{}"), mat.rows(), mat.cols());
    }
    auto const svd = mat.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
    this->vals = svd.singularValues();
    this->U = svd.matrixU();
    this->V = svd.matrixV();
  }
}

template struct SVD<float>;
template struct SVD<Cx>;
}
