#include "decomp.h"
#include "tensorOps.h"

// #include <Eigen/Eigenvalues>
#include <Eigen/SVD>

Cx5 LowRankKernels(Cx5 const &mIn, float const thresh)
{
  Index const kSz = mIn.dimension(0) * mIn.dimension(1) * mIn.dimension(2) * mIn.dimension(3);
  Index const nK = mIn.dimension(4);
  Eigen::Map<Eigen::MatrixXcf const> m(mIn.data(), kSz, nK);
  Log::Print(FMT_STRING("SVD Kernel Size {} Kernels {}"), kSz, nK);
  auto const svd = m.transpose().bdcSvd(Eigen::ComputeThinV);
  Eigen::ArrayXf const vals = svd.singularValues();
  Index const nRetain = (vals > (vals[0] * thresh)).cast<int>().sum();
  Log::Print(FMT_STRING("Retaining {} kernels"), nRetain);
  Cx5 out(mIn.dimension(0), mIn.dimension(1), mIn.dimension(2), mIn.dimension(3), nRetain);
  Eigen::Map<Eigen::MatrixXcf> lr(out.data(), kSz, nRetain);
  lr = svd.matrixV().leftCols(nRetain).conjugate();
  return out;
}

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

SVDResult SVD(Eigen::ArrayXXf const &mat)
{
  auto const svd = mat.matrix().bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
  return SVDResult{.v = svd.matrixV(), .u = svd.matrixU(), .vals = svd.singularValues()};
}
