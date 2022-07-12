#include "llr.h"

#include "decomp.h"
#include "tensorOps.h"
#include "threads.h"
#include <random>

namespace rl {
Index PatchClamp(Index const ii, Index const pSz, Index const dimSz)
{
  Index const hSz = pSz / 2;
  if (ii < hSz) {
    return ii;
  } else if (ii > (dimSz - hSz)) {
    return ii - (dimSz - pSz);
  } else {
    return hSz;
  }
}

Cx4 llr_sliding(Cx4 const &img, float const λ, Index const pSz)
{
  Index const K = img.dimension(0);
  Log::Print(FMT_STRING("LLR regularization patch size {} lambda {}"), pSz, λ);
  Cx4 lr(img.dimensions());
  lr.setZero();

  auto zTask = [&](Index const iz) {
    for (Index iy = 0; iy < img.dimension(2); iy++) {
      for (Index ix = 0; ix < img.dimension(1); ix++) {
        Index const stx = std::min(std::max(0L, ix - pSz / 2), img.dimension(1) - pSz);
        Index const sty = std::min(std::max(0L, iy - pSz / 2), img.dimension(2) - pSz);
        Index const stz = std::min(std::max(0L, iz - pSz / 2), img.dimension(3) - pSz);
        Cx4 patchTensor = img.slice(Sz4{0, stx, sty, stz}, Sz4{K, pSz, pSz, pSz});
        auto patch = CollapseToMatrix(patchTensor);
        auto const svd = SVD<Cx>(patch, true, false);
        // Soft-threhold svals
        Eigen::VectorXf s = svd.vals * (svd.vals.abs() - λ) / svd.vals.abs();
        s = (s.array() > λ).select(s, 0.f);
        patch = (svd.U * s.asDiagonal() * svd.V.adjoint()).transpose();
        lr.chip<3>(iz).chip<2>(iy).chip<1>(ix) = patchTensor.chip<3>(PatchClamp(iz, pSz, img.dimension(3)))
                                                   .chip<2>(PatchClamp(iy, pSz, img.dimension(2)))
                                                   .chip<1>(PatchClamp(ix, pSz, img.dimension(1)));
      }
    }
  };
  auto const now = Log::Now();
  Threads::For(zTask, 0, img.dimension(3), "LLR");
  Log::Print(FMT_STRING("LLR Regularization took {}"), Log::ToNow(now));
  return lr;
}

Cx4 llr_patch(Cx4 const &x, float const l, Index const p)
{
  std::array<Index, 3> nP, shift;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> int_dist(0, p - 1);
  for (Index ii = 0; ii < 3; ii++) {
    if (x.dimension(ii + 1) % p != 0) {
      Log::Fail(FMT_STRING("Patch size {} does not evenly divide {} (dimension {})"), p, x.dimension(ii + 1), ii);
    }
    nP[ii] = (x.dimension(ii + 1) / p) - 1;
    shift[ii] = int_dist(gen);
  }
  Index const K = x.dimension(0);
  Index const pSz = p * p * p;
  Cx4 lr(x.dimensions());
  lr.setZero();
  auto zTask = [&](Index const lo, Index const hi) {
    for (Index iz = lo; iz < hi; iz++) {
      for (Index iy = 0; iy < nP[1]; iy++) {
        for (Index ix = 0; ix < nP[0]; ix++) {
          Cx4 px = x.slice(Sz4{0, ix * p + shift[0], iy * p + shift[1], iz * p + shift[2]}, Sz4{K, p, p, p});
          Eigen::Map<Eigen::MatrixXcf> patch(px.data(), K, pSz);
          auto const svd = patch.transpose().bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
          // Soft-threhold svals
          Eigen::ArrayXf s = svd.singularValues();
          float const sl = s(0) * l;
          s = s * (s.abs() - sl) / s.abs();
          s = (s > sl).select(s, 0.f);
          patch.transpose() = svd.matrixU() * s.matrix().asDiagonal() * svd.matrixV().adjoint();
          lr.slice(Sz4{0, ix * p + shift[0], iy * p + shift[1], iz * p + shift[2]}, Sz4{K, p, p, p}) = px;
        }
      }
    }
  };
  auto const now = Log::Now();
  zTask(0, nP[2]);
  // Threads::RangeFor(zTask, nP[2]);
  Log::Print(FMT_STRING("LLR Regularization took {}"), Log::ToNow(now));
  return lr;
}
}
