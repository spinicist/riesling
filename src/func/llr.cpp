#include "llr.hpp"

#include "algo/decomp.hpp"
#include "tensorOps.hpp"
#include "threads.hpp"
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

LLR::LLR(float l, Index p, bool s)
  : Functor<Cx4>()
  , patchSize{p}
  , λ{l * std::sqrtf(p)}
  , sliding{s}
{
}

auto LLR::operator()(Cx4 const &x) const -> Cx4
{
  if (sliding) {
    return applySliding(x);
  } else {
    return applyFixed(x);
  }
}

auto LLR::applySliding(Cx4 const &img) const -> Cx4
{
  Index const K = img.dimension(0);
  Log::Print(FMT_STRING("LLR regularization patch size {} lambda {}"), patchSize, λ);
  Cx4 lr(img.dimensions());
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
        Eigen::VectorXf s = svd.vals * (svd.vals.abs() - λ) / svd.vals.abs();
        s = (s.array() > λ).select(s, 0.f);
        patch = (svd.U * s.asDiagonal() * svd.V.adjoint()).transpose();
        for (Index ii = 0; ii < K; ii++) {
          lr(ii, ix, iy, iz) = patchTensor(ii, PatchClamp(ix, patchSize, img.dimension(1)), PatchClamp(iy, patchSize, img.dimension(2)), PatchClamp(iz, patchSize, img.dimension(3)));
        }
        // lr(0, ix, iy, iz) = PatchClamp(ix, patchSize, img.dimension(1));
        // lr(1, ix, iy, iz) = PatchClamp(iy, patchSize, img.dimension(2));
        // lr(2, ix, iy, iz) = PatchClamp(iz, patchSize, img.dimension(3));
        // lr(3, ix, iy, iz) = 0.f;
      }
    }
  };
  auto const now = Log::Now();
  Threads::For(zTask, img.dimension(3), "LLR");
  Log::Print(FMT_STRING("LLR Regularization took {}"), Log::ToNow(now));
  return lr;
}

auto LLR::applyFixed(Cx4 const &x) const -> Cx4
{
  std::array<Index, 3> nP, shift;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> int_dist(0, patchSize - 1);
  for (Index ii = 0; ii < 3; ii++) {
    if (x.dimension(ii + 1) % patchSize != 0) {
      Log::Fail(
        FMT_STRING("Patch size {} does not evenly divide {} (dimension {})"), patchSize, x.dimension(ii + 1), ii);
    }
    nP[ii] = (x.dimension(ii + 1) / patchSize) - 1;
    shift[ii] = int_dist(gen);
  }
  Index const K = x.dimension(0);
  Index const p3 = patchSize * patchSize * patchSize;
  Cx4 lr(x.dimensions());
  lr.setZero();
  auto zTask = [&](Index const lo, Index const hi) {
    for (Index iz = lo; iz < hi; iz++) {
      for (Index iy = 0; iy < nP[1]; iy++) {
        for (Index ix = 0; ix < nP[0]; ix++) {
          Cx4 px = x.slice(
            Sz4{0, ix * patchSize + shift[0], iy * patchSize + shift[1], iz * patchSize + shift[2]},
            Sz4{K, patchSize, patchSize, patchSize});
          Eigen::Map<Eigen::MatrixXcf> patch(px.data(), K, p3);
          auto const svd = SVD<Cx>(patch, true, false);
          // Soft-threhold svals
          Eigen::ArrayXf s = svd.vals;
          float const sl = s(0) * λ;
          s = s * (s.abs() - sl) / s.abs();
          s = (s > sl).select(s, 0.f);
          patch.transpose() = svd.U * s.matrix().asDiagonal() * svd.V.adjoint();
          lr.slice(
            Sz4{0, ix * patchSize + shift[0], iy * patchSize + shift[1], iz * patchSize + shift[2]},
            Sz4{K, patchSize, patchSize, patchSize}) = px;
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
} // namespace rl
