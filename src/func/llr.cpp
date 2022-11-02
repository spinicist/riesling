#include "llr.hpp"

#include "algo/decomp.hpp"
#include "tensorOps.hpp"
#include "threads.hpp"
#include <cmath>
#include <random>

namespace rl {
Index PatchClamp(Index const ii, Index const patchSize, Index const dimSz)
{
  Index const hSz = (patchSize + 1) / 2;
  if (ii < hSz) {
    return ii;
  } else if (ii > (dimSz - hSz)) {
    return ii - (dimSz - patchSize);
  } else {
    return hSz;
  }
}

LLR::LLR(Index p, bool s)
  : Prox<Cx4>()
  , patchSize{p}
  , sliding{s}
{
}

auto LLR::operator()(float const λ, Eigen::TensorMap<Cx4 const> x) const -> Eigen::TensorMap<Cx4>
{
  if (sliding) {
    return applySliding(λ * std::sqrt(patchSize), x);
  } else {
    return applyFixed(λ * std::sqrt(patchSize), x);
  }
}

auto LLR::applySliding(float const λ, Eigen::TensorMap<Cx4 const> img) const -> Eigen::TensorMap<Cx4>
{
  Index const K = img.dimension(0);
  Log::Print(FMT_STRING("LLR regularization patch size {} lambda {}"), patchSize, λ);
  static Cx4 lr(img.dimensions());
  lr.setZero();

  auto zTask = [&](Index const iz) {
    for (Index iy = 0; iy < img.dimension(2); iy++) {
      for (Index ix = 0; ix < img.dimension(1); ix++) {
        Index const stx = std::min(std::max(0L, ix - (patchSize + 1) / 2), img.dimension(1) - patchSize);
        Index const sty = std::min(std::max(0L, iy - (patchSize + 1) / 2), img.dimension(2) - patchSize);
        Index const stz = std::min(std::max(0L, iz - (patchSize + 1) / 2), img.dimension(3) - patchSize);
        Cx4 patchTensor = img.slice(Sz4{0, stx, sty, stz}, Sz4{K, patchSize, patchSize, patchSize});
        auto patch = CollapseToMatrix(patchTensor);
        auto const svd = SVD<Cx>(patch, true, false);
        // Soft-threhold svals
        Eigen::VectorXf const s = (svd.vals.abs() > λ).select(svd.vals * (svd.vals.abs() - λ) / svd.vals.abs(), 0.f);
        patch = (svd.U * s.asDiagonal() * svd.V.adjoint()).transpose();
        for (Index ii = 0; ii < K; ii++) {
          lr(ii, ix, iy, iz) = patchTensor(
            ii,
            PatchClamp(ix, patchSize, img.dimension(1)),
            PatchClamp(iy, patchSize, img.dimension(2)),
            PatchClamp(iz, patchSize, img.dimension(3)));
        }
      }
    }
  };
  auto const now = Log::Now();
  Threads::For(zTask, img.dimension(3), "LLR");
  Log::Print(FMT_STRING("LLR Regularization took {}"), Log::ToNow(now));
  return lr;
}

auto LLR::applyFixed(float const λ, Eigen::TensorMap<Cx4 const> x) const -> Eigen::TensorMap<Cx4>
{
  std::array<Index, 3> nP, shift;
  std::random_device rd;
  std::mt19937 gen(rd());
  for (Index ii = 0; ii < 3; ii++) {
    auto const d = x.dimension(ii + 1);
    nP[ii] = ((d - 1) / patchSize) - 1;
    std::uniform_int_distribution<> int_dist(0, d - nP[ii] * patchSize);
    shift[ii] = int_dist(gen);
  }
  Index const K = x.dimension(0);
  static Cx4 lr(x.dimensions());
  lr.setZero();
  auto zTask = [&](Index const iz) {
    for (Index iy = 0; iy < nP[1]; iy++) {
      for (Index ix = 0; ix < nP[0]; ix++) {
        Index const stx = ix * patchSize + shift[0];
        Index const sty = iy * patchSize + shift[1];
        Index const stz = iz * patchSize + shift[2];
        Cx4 patchTensor = x.slice(Sz4{0, stx, sty, stz}, Sz4{K, patchSize, patchSize, patchSize});
        auto patch = CollapseToMatrix(patchTensor);
        auto const svd = SVD<Cx>(patch, true, false);
        // Soft-threhold svals
        Eigen::VectorXf const s = (svd.vals.abs() > λ).select(svd.vals * (svd.vals.abs() - λ) / svd.vals.abs(), 0.f);
        patch = (svd.U * s.asDiagonal() * svd.V.adjoint()).transpose();
        lr.slice(Sz4{0, stx, sty, stz}, Sz4{K, patchSize, patchSize, patchSize}) = patchTensor;
      }
    }
  };
  auto const now = Log::Now();
  Threads::For(zTask, nP[2], "LLR");
  Log::Print(FMT_STRING("LLR Regularization took {}"), Log::ToNow(now));
  return lr;
}
} // namespace rl
