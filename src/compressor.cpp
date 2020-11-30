#include "compressor.h"

#include <Eigen/Dense>

Compressor::Compressor(Cx3 const &ks, long const nc, Log &log)
    : log_{log}
{
  Eigen::Map<Eigen::MatrixXcf const> const km(
      ks.data(), ks.dimension(0), ks.dimension(1) * ks.dimension(2));
  Eigen::MatrixXcf gramian = (km.conjugate() * km.transpose()) / pow(km.norm(), 2);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcf> eig(gramian);
  Eigen::ArrayXf vals = eig.eigenvalues().reverse().array().abs();
  vals /= vals.abs().sum();
  long const n = std::min(nc, ks.dimension(0));
  log.info(
      FMT_STRING("PCA Compression Retaining {} virtual coils, total energy {}%"),
      n,
      100.f * vals.head(n).abs().sum());
  Eigen::MatrixXcf vecs = eig.eigenvectors().rowwise().reverse();
  psi_ = vecs.leftCols(n);
}

long Compressor::out_channels() const
{
  return psi_.cols();
}

void Compressor::compress(Cx3 const &source, Cx3 &dest)
{
  if (psi_.size() != 0) {
    assert(source.dimension(0) == psi_.cols());
    assert(dest.dimension(0) == psi_.rows());
    log_.info(FMT_STRING("Applying coil compression"));
    Eigen::Map<Eigen::MatrixXcf const> const sourcemat(
        source.data(), source.dimension(0), source.dimension(1) * source.dimension(2));
    Eigen::Map<Eigen::MatrixXcf> destmat(
        dest.data(), dest.dimension(0), dest.dimension(1) * dest.dimension(2));
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
    long const samples = source.dimension(1) * source.dimension(2);
    long const s_sz = samples * source.dimension(0);
    long const d_sz = samples * dest.dimension(0);
    for (long iv = 0; iv < source.dimension(3); iv++) {
      auto const sd = source.data() + s_sz * iv;
      auto const dd = dest.data() + d_sz * iv;
      Eigen::Map<Eigen::MatrixXcf const> const sourcemat(sd, source.dimension(0), samples);
      Eigen::Map<Eigen::MatrixXcf> destmat(dd, dest.dimension(0), samples);
      destmat.noalias() = psi_.transpose() * sourcemat;
    }
  } else {
    dest = source;
  }
}