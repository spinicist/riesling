#include "llr.hpp"

#include "algo/decomp.hpp"
#include "tensorOps.hpp"
#include "threads.hpp"
#include "log.hpp"

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

LLR::LLR(float const l, Index const p, Index const w, Sz4 const s)
  : Prox<Cx>()
  , λ{l}
  , patchSize{p}
  , windowSize{w}
  , shape{s}
{
  Log::Print("Locally Low-Rank λ {} Patch {} Window {}", λ, patchSize, windowSize);
}

void LLR::apply(float const α, CMap const &xin, Map &zin) const
{
  Eigen::TensorMap<Cx4 const> x(xin.data(), shape);
  Sz3 nWindows, shift;
  std::random_device rd;
  std::mt19937 gen(rd());
  for (Index ii = 0; ii < 3; ii++) {
    auto const d = x.dimension(ii + 1);
    nWindows[ii] = ((d - 1) / windowSize) - 1;
    std::uniform_int_distribution<> int_dist(0, d - nWindows[ii] * windowSize);
    shift[ii] = int_dist(gen);
  }
  Index const K = x.dimension(0);
  Sz4 const szP{K, patchSize, patchSize, patchSize};
  Sz4 const szW{K, windowSize, windowSize, windowSize};
  Index const inset = (patchSize - windowSize) / 2;
  Eigen::TensorMap<Cx4> z(zin.data(), shape);
  z.setZero();
  float const realλ = λ * α;
  auto zTask = [&](Index const iz) {
    for (Index iy = 0; iy < nWindows[1]; iy++) {
      for (Index ix = 0; ix < nWindows[0]; ix++) {
        Sz3 ind{ix, iy, iz};
        Sz4 stP, stW, stW2;
        stP[0] = stW[0] = stW2[0] = 0;
        for (Index ii = 0; ii < 3; ii++) {
          stW[ii + 1] = ind[ii] * windowSize + shift[ii];
          stP[ii + 1] = std::clamp(stW[ii + 1] - inset, 0L, x.dimension(ii + 1) - patchSize);
          stW2[ii + 1] = stW[ii + 1] - stP[ii + 1];
        }
        Cx4 patchTensor = x.slice(stP, szP);
        auto patch = CollapseToMatrix(patchTensor);
        auto const svd = SVD<Cx>(patch, true, false);
        // Soft-threhold svals
        Eigen::VectorXf const s = (svd.vals.abs() > realλ).select(svd.vals * (svd.vals.abs() - realλ) / svd.vals.abs(), 0.f);
        patch = (svd.U * s.asDiagonal() * svd.V.adjoint()).transpose();
        z.slice(stW, szW) = patchTensor.slice(stW2, szW);
      }
    }
  };
  Threads::For(zTask, nWindows[2], "LLR");
  Log::Print("LLR α {} λ {} t {} |x| {} |z| {}", α, λ, realλ, Norm(x), Norm(z));
}

} // namespace rl
