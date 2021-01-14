#include "compressor.h"

#include "tensorOps.h"

Compressor::Compressor(Cx3 const &ks, long const nc, Log &log)
    : log_{log}
{
  auto const km = CollapseToMatrix(ks);
  auto const dm = km.colwise() - km.rowwise().mean();
  Eigen::MatrixXcf gramian = (dm.conjugate() * dm.transpose()) / (km.rows() - 1);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcf> eig(gramian);
  Eigen::ArrayXf vals = eig.eigenvalues().reverse().array().abs();
  vals /= vals.abs().sum();
  long const n = std::min(nc, ks.dimension(0));
  log.info(
      FMT_STRING("PCA Compression Retaining {} virtual coils, total energy {}%"),
      n,
      100.f * vals.head(n).sum());
  psi_ = eig.eigenvectors().rightCols(n).rowwise().reverse().transpose();
}

long Compressor::out_channels() const
{
  return psi_.rows();
}

void Compressor::compress(Cx3 const &source, Cx3 &dest)
{
  if (psi_.size() != 0) {
    assert(source.dimension(0) == psi_.rows());
    assert(dest.dimension(0) == psi_.cols());
    log_.info(FMT_STRING("Applying coil compression"));
    auto const sourcemat = CollapseToMatrix(source);
    auto destmat = CollapseToMatrix(dest);
    destmat.noalias() = psi_ * sourcemat;
  } else {
    dest = source;
  }
}

void Compressor::compress(Cx4 const &source, Cx4 &dest)
{
  if (psi_.size() != 0) {
    assert(source.dimension(1) == dest.dimension(1));
    assert(source.dimension(2) == dest.dimension(2));
    assert(source.dimension(3) == dest.dimension(3));
    assert(source.dimension(0) == psi_.rows());
    assert(dest.dimension(0) == psi_.cols());
    log_.info(FMT_STRING("Applying coil compression"));
    auto const sourcemat = CollapseToMatrix(source);
    auto destmat = CollapseToMatrix(dest);
    destmat.noalias() = psi_ * sourcemat;
  } else {
    dest = source;
  }
}