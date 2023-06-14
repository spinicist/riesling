#include "llr.hpp"

#include "algo/decomp.hpp"
#include "log.hpp"
#include "patches.hpp"
#include "tensorOps.hpp"

namespace rl {

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
  Eigen::TensorMap<Cx4> z(zin.data(), shape);
  float const realλ = λ * α * std::sqrt(patchSize*patchSize*patchSize);

  auto softLLR = [realλ] (Cx4 const &xp) {
        Eigen::MatrixXcf patch = CollapseToMatrix(xp);
        auto const svd = SVD<Cx>(patch, true, false);
        // Soft-threhold svals
        Eigen::VectorXf const s = (svd.vals.abs() > realλ).select(svd.vals * (svd.vals.abs() - realλ) / svd.vals.abs(), 0.f);
        patch = (svd.U * s.asDiagonal() * svd.V.adjoint()).transpose();
        Cx4 yp = Tensorfy(patch, xp.dimensions());
        return yp;
      };
  Patches(patchSize, windowSize, softLLR, x, z);
  Log::Print("LLR α {} λ {} t {} |x| {} |z| {}", α, λ, realλ, Norm(x), Norm(z));
}

} // namespace rl
