#include "llr.hpp"

#include "../algo/decomp.hpp"
#include "../log/log.hpp"
#include "../patches.hpp"
#include "../tensors.hpp"

namespace rl::Proxs {

LLR::LLR(float const l, Index const p, Index const w, bool const doShift, Sz5 const s)
  : Prox(Product(s))
  , λ{l}
  , patchSize{p}
  , windowSize{w}
  , shape{s}
  , shift{doShift}
{
  /* λ needs to be scaled to work across block-sizes etc.
   * This is the scaling in BART which is taken from Ong 2016 Beyond Low Rank + Sparse: Multiscale Low Rank Matrix Decomposition
   */
  Index const M = p * p * p;
  Index const N = Product(LastN<2>(s));
  Index const B = Product(FirstN<3>(s)) / (M * N);
  λ *= (std::sqrt(M) + std::sqrt(N) + std::sqrt(std::log(B * std::min(M, N))));
  Log::Print("Prox", "Locally Low-Rank λ {} Scaled λ {} Patch {} Window {}", l, λ, patchSize, windowSize);
}

void LLR::apply(float const α, CMap xin, Map zin) const
{
  Cx5CMap     x(xin.data(), shape);
  Cx5Map      z(zin.data(), shape);
  float const realλ = λ * α;

  auto softLLR = [realλ](Cx5 const &xp) {
    Eigen::MatrixXcf patch = CollapseToMatrix<Cx5, 3>(xp);
    auto const       svd = SVD<Cx>(patch);
    // Soft-threhold svals
    Eigen::VectorXf const s = (svd.S.abs() > realλ).select(svd.S * (svd.S.abs() - realλ) / svd.S.abs(), 0.f);
    patch = (svd.U * s.asDiagonal() * svd.V.adjoint());
    Cx5 yp = AsTensorMap(patch, xp.dimensions());
    return yp;
  };
  Patches(patchSize, windowSize, shift, softLLR, x, z);
  Log::Debug("Prox", "LLR α {} λ {} t {} |x| {} |z| {}", α, λ, realλ, Norm<true>(x), Norm<true>(z));
}

void LLR::conj(float const α, CMap xin, Map zin) const
{
  Cx5CMap     x(xin.data(), shape);
  Cx5Map      z(zin.data(), shape);
  /* Amazingly, this doesn't depend on α. Maths is based. */
  auto projLLR = [λ = this->λ](Cx5 const &xp) {
    Eigen::MatrixXcf patch = CollapseToMatrix<Cx5, 3>(xp);
    auto const       svd = SVD<Cx>(patch);
    Eigen::VectorXf const s = (svd.S.abs() > λ).select(λ * svd.S / svd.S.abs(), svd.S);
    patch = (svd.U * s.asDiagonal() * svd.V.adjoint());
    Cx5 yp = AsTensorMap(patch, xp.dimensions());
    return yp;
  };
  Patches(patchSize, windowSize, shift, projLLR, x, z);
  Log::Debug("Prox", "LLR α {} λ {} |x| {} |z| {}", α, λ, Norm<true>(x), Norm<true>(z));
}

} // namespace rl::Proxs
