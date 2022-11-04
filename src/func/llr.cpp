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

LLR::LLR(float const l, Index p, Index w)
  : Prox<Cx4>()
  , λ{l}
  , patchSize{p}
  , windowSize{w}
{
}

auto LLR::operator()(float const α, Eigen::TensorMap<Cx4 const> x) const -> Cx4
{
  Sz3 nP, shift;
  std::random_device rd;
  std::mt19937 gen(rd());
  for (Index ii = 0; ii < 3; ii++) {
    auto const d = x.dimension(ii + 1);
    nP[ii] = ((d - 1) / windowSize) - 1;
    std::uniform_int_distribution<> int_dist(0, d - nP[ii] * windowSize);
    shift[ii] = int_dist(gen);
  }
  Index const K = x.dimension(0);
  Sz4 const szP{K, patchSize, patchSize, patchSize};
  Sz4 const szW{K, windowSize, windowSize, windowSize};
  Index const inset = (patchSize - windowSize) / 2;
  Cx4 lr = x;
  float const realλ = λ * α;
  Log::Print<Log::Level::High>(FMT_STRING("LLR λ {} Patch-size {} Window-size {}"), realλ, patchSize, windowSize);
  auto zTask = [&](Index const iz) {
    for (Index iy = 0; iy < nP[1]; iy++) {
      for (Index ix = 0; ix < nP[0]; ix++) {
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
        lr.slice(stW, szW) = patchTensor.slice(stW2, szW);
      }
    }
  };
  Threads::For(zTask, nP[2], "LLR");
  return lr;
}

} // namespace rl
