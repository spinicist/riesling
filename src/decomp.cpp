#include "decomp.h"

#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

Cx5 LowRankKernels(Cx5 const &mIn, float const thresh, Log const &log)
{
  long const kSz = mIn.dimension(0) * mIn.dimension(1) * mIn.dimension(2) * mIn.dimension(3);
  long const nK = mIn.dimension(4);
  Eigen::Map<Eigen::MatrixXcf const> m(mIn.data(), kSz, nK);
  log.info(FMT_STRING("SVD Kernel Size {} Kernels {}"), kSz, nK);
  auto const svd = m.transpose().bdcSvd(Eigen::ComputeThinV);
  Eigen::ArrayXf const vals = svd.singularValues();
  long const nRetain = (vals > (vals[0] * thresh)).cast<int>().sum();
  log.info(FMT_STRING("Retaining {} kernels"), nRetain);
  Cx5 out(mIn.dimension(0), mIn.dimension(1), mIn.dimension(2), mIn.dimension(3), nRetain);
  Eigen::Map<Eigen::MatrixXcf> lr(out.data(), kSz, nRetain);
  lr = svd.matrixV().leftCols(nRetain).conjugate();
  return out;
}

Cx2 Covariance(Cx2 const &data)
{
  Cx const scale(1.f / data.dimension(1));
  return data.conjugate().contract(data, Eigen::IndexPairList<Eigen::type2indexpair<1, 1>>()) *
         scale;
}

void PCA(Cx2 const &gramIn, Cx2 &vecIn, R1 &valIn, Log const &log)
{
  Eigen::Map<Eigen::MatrixXcf const> gram(gramIn.data(), gramIn.dimension(0), gramIn.dimension(1));
  if (gram.adjoint() != gram) {
    log.fail("Gram Matrix was not self-adjoint");
  }
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcf> es;
  es.compute(gram);
  Eigen::Map<Eigen::MatrixXcf> vec(vecIn.data(), vecIn.dimension(0), vecIn.dimension(1));
  Eigen::Map<Eigen::VectorXf> val(valIn.data(), valIn.dimension(0));
  assert(vec.rows() == gram.rows());
  assert(vec.cols() == gram.cols());
  assert(val.rows() == gram.rows());
  vec = es.eigenvectors();
  val = es.eigenvalues();
}