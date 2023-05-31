#include "llr.hpp"

#include "algo/decomp.hpp"
#include "log.hpp"
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
    nWindows[ii] = (d / windowSize) + 2;
    std::uniform_int_distribution<> int_dist(0, windowSize - 1);
    shift[ii] = int_dist(gen);
  }
  Log::Print<Log::Level::Debug>("LLR Shift {}", shift);
  Index const K = x.dimension(0);
  Sz4 const szP{K, patchSize, patchSize, patchSize};
  Index const inset = (patchSize - windowSize) / 2;
  Eigen::TensorMap<Cx4> z(zin.data(), shape);
  z.setConstant(25.f);
  float const realλ = λ * α * std::sqrt(patchSize*patchSize*patchSize);
  auto zTask = [&](Index const iz) {
    for (Index iy = 0; iy < nWindows[1]; iy++) {
      for (Index ix = 0; ix < nWindows[0]; ix++) {
        Sz3 ind{ix - 1, iy - 1, iz - 1};
        Sz4 stP, stW, stW2, szW;
        stP[0] = stW[0] = stW2[0] = 0;
        szW[0] = K;
        bool empty = false;
        for (Index ii = 0; ii < 3; ii++) {
          Index const d = x.dimension(ii + 1);
          Index const st = ind[ii] * windowSize + shift[ii];
          stW[ii + 1] = std::max(st, 0L);
          szW[ii + 1] = windowSize + std::min({st, 0L, d - stW[ii + 1] - windowSize});
          if (szW[ii + 1] < 1) {
            empty = true;
            break;
          }
          stP[ii + 1] = std::clamp(st - inset, 0L, d - patchSize);
          stW2[ii + 1] = stW[ii + 1] - stP[ii + 1];
        }
        if (empty) {
          continue;
        }
        Cx4 patchTensor = x.slice(stP, szP);
        Eigen::MatrixXcf patch = CollapseToMatrix(patchTensor);
        auto const svd = SVD<Cx>(patch, true, false);
        // Soft-threhold svals
        Eigen::VectorXf const s = (svd.vals.abs() > realλ).select(svd.vals * (svd.vals.abs() - realλ) / svd.vals.abs(), 0.f);
        patch = (svd.U * s.asDiagonal() * svd.V.adjoint()).transpose();
        patchTensor = Tensorfy(patch, szP);
        z.slice(stW, szW) = patchTensor.slice(stW2, szW);
      }
    }
  };
  Threads::For(zTask, nWindows[2], "LLR");
  Log::Print("LLR α {} λ {} t {} |x| {} |z| {}", α, λ, realλ, Norm(x), Norm(z));
}

} // namespace rl
