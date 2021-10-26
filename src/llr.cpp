#include "llr.h"

#include "threads.h"
#include <Eigen/SVD>
#include <random>

Cx4 llr(Cx4 const &x, float const l, long const p, Log &log)
{
  long const K = x.dimension(0);
  log.info("LLR regularization patch size {} lamdba {}", p, l);
  Cx4 lr(x.dimensions());
  lr.setZero();

  auto zTask = [&](long const lo, long const hi) {
    for (long iz = lo; iz < hi; iz++) {
      for (long iy = 0; iy < x.dimension(2) - p; iy++) {
        for (long ix = 0; ix < x.dimension(1) - p; ix++) {
          Cx4 px = x.slice(Sz4{0, ix, iy, iz}, Sz4{K, p, p, p});
          Eigen::Map<Eigen::MatrixXcf> patch(px.data(), K, p * p * p);
          auto const svd = patch.transpose().bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
          // Soft-threhold svals
          Eigen::ArrayXf s = svd.singularValues();
          float const sl = s.sum() * l;
          s = s * (s.abs() - sl) / s.abs();
          s = (s > sl).select(s, 0.f);
          patch.transpose() = svd.matrixU() * s.matrix().asDiagonal() * svd.matrixV().adjoint();
          lr.chip(iz + p / 2, 3).chip(iy + p / 2, 2).chip(ix + p / 2, 1) =
              px.chip(p / 2, 3).chip(p / 2, 2).chip(p / 2, 1);
        }
      }
    }
  };
  auto const now = log.now();
  Threads::RangeFor(zTask, 0, x.dimension(3) - p);
  log.info("LLR Regularization took {}", log.toNow(now));
  return lr;
}

Cx4 llr_patch(Cx4 const &x, float const l, long const p, Log &log)
{
  std::array<long, 3> nP, shift;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> int_dist(0, p - 1);
  for (long ii = 0; ii < 3; ii++) {
    if (x.dimension(ii + 1) % p != 0) {
      Log::Fail(
          FMT_STRING("Patch size {} does not evenly divide {} (dimension {})"),
          p,
          x.dimension(ii + 1),
          ii);
    }
    nP[ii] = (x.dimension(ii + 1) / p) - 1;
    shift[ii] = int_dist(gen);
  }
  fmt::print("Offset: {}\n", fmt::join(shift, ","));
  long const K = x.dimension(0);
  long const pSz = p * p * p;
  Cx4 lr(x.dimensions());
  lr.setZero();
  auto zTask = [&](long const lo, long const hi) {
    for (long iz = lo; iz < hi; iz++) {
      for (long iy = 0; iy < nP[1]; iy++) {
        for (long ix = 0; ix < nP[0]; ix++) {
          Cx4 px = x.slice(
              Sz4{0, ix * p + shift[0], iy * p + shift[1], iz * p + shift[2]}, Sz4{K, p, p, p});
          Eigen::Map<Eigen::MatrixXcf> patch(px.data(), K, pSz);
          auto const svd = patch.transpose().bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
          // Soft-threhold svals
          Eigen::ArrayXf s = svd.singularValues();
          float const sl = s(0) * l;
          s = s * (s.abs() - sl) / s.abs();
          s = (s > sl).select(s, 0.f);
          patch.transpose() = svd.matrixU() * s.matrix().asDiagonal() * svd.matrixV().adjoint();
          lr.slice(
              Sz4{0, ix * p + shift[0], iy * p + shift[1], iz * p + shift[2]}, Sz4{K, p, p, p}) =
              px;
        }
      }
    }
  };
  auto const now = log.now();
  zTask(0, nP[2]);
  // Threads::RangeFor(zTask, nP[2]);
  log.info("LLR Regularization took {}", log.toNow(now));
  return lr;
}