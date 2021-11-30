#include "decomp.h"

// #include <Eigen/Eigenvalues>
#include <Eigen/SVD>

Cx5 LowRankKernels(Cx5 const &mIn, float const thresh, Log const &log)
{
  Index const kSz = mIn.dimension(0) * mIn.dimension(1) * mIn.dimension(2) * mIn.dimension(3);
  Index const nK = mIn.dimension(4);
  Eigen::Map<Eigen::MatrixXcf const> m(mIn.data(), kSz, nK);
  log.info(FMT_STRING("SVD Kernel Size {} Kernels {}"), kSz, nK);
  auto const svd = m.transpose().bdcSvd(Eigen::ComputeThinV);
  Eigen::ArrayXf const vals = svd.singularValues();
  Index const nRetain = (vals > (vals[0] * thresh)).cast<int>().sum();
  log.info(FMT_STRING("Retaining {} kernels"), nRetain);
  Cx5 out(mIn.dimension(0), mIn.dimension(1), mIn.dimension(2), mIn.dimension(3), nRetain);
  Eigen::Map<Eigen::MatrixXcf> lr(out.data(), kSz, nRetain);
  lr = svd.matrixV().leftCols(nRetain).conjugate();
  return out;
}

void PCA(Cx2 const &dataIn, Cx2 &vecIn, R1 &valIn, Log const &log)
{
  Eigen::Map<Eigen::MatrixXcf const> data(dataIn.data(), dataIn.dimension(0), dataIn.dimension(1));
  Eigen::Map<Eigen::MatrixXcf> vecs(vecIn.data(), vecIn.dimension(0), vecIn.dimension(1));
  Eigen::Map<Eigen::VectorXf> vals(valIn.data(), valIn.dimension(0));
  assert(vecs.rows() == data.rows());
  assert(vecs.cols() == data.rows());
  assert(vals.rows() == data.rows());
  auto const svd = data.transpose().bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
  Eigen::MatrixXcf const V = svd.matrixV();
  vecs = V;
  vals = svd.singularValues().array().sqrt();
}