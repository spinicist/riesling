#include "llr.hpp"

#include "algo/decomp.hpp"
#include "log.hpp"
#include "patches.hpp"
#include "tensors.hpp"

namespace rl::Proxs {

LLR::LLR(float const l, Index const p, Index const w, bool const doShift, Sz4 const s)
  : Prox<Cx>(Product(s))
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
  Index const N = s[0];
  Index const B = Product(LastN<3>(s)) / (M * N);
  λ *= (std::sqrt(M) + std::sqrt(N) + std::sqrt(std::log(B * std::min(M, N))));
  Log::Print("Locally Low-Rank λ {} Scaled λ {} Patch {} Window {}", l, λ, patchSize, windowSize);
}

void LLR::apply(float const α, CMap const &xin, Map &zin) const
{
  Eigen::TensorMap<Cx4 const> x(xin.data(), shape);
  Eigen::TensorMap<Cx4>       z(zin.data(), shape);
  float const                 realλ = λ * α;

  auto softLLR = [realλ](Cx4 const &xp) {
    Eigen::MatrixXcf patch = CollapseToMatrix(xp);
    auto const       svd = SVD<Cx>(patch.transpose());
    // Soft-threhold svals
    Eigen::VectorXf const s = (svd.S.abs() > realλ).select(svd.S * (svd.S.abs() - realλ) / svd.S.abs(), 0.f);
    patch = (svd.U * s.asDiagonal() * svd.V.adjoint()).transpose();
    Cx4 yp = Tensorfy(patch, xp.dimensions());
    return yp;
  };
  Patches(patchSize, windowSize, shift, softLLR, x, z);
  Log::Debug("LLR α {} λ {} t {} |x| {} |z| {}", α, λ, realλ, Norm(x), Norm(z));
}

void LLR::apply(std::shared_ptr<Op> const α, CMap const &xin, Map &zin) const
{
  if (auto realα = std::dynamic_pointer_cast<Ops::DiagScale<Cx>>(α)) {
    Eigen::TensorMap<Cx4 const> x(xin.data(), shape);
    Eigen::TensorMap<Cx4>       z(zin.data(), shape);
    float const                 realλ = λ * realα->scale * std::sqrt(patchSize * patchSize * patchSize);

    auto softLLR = [realλ](Cx4 const &xp) {
      Eigen::MatrixXcf patch = CollapseToMatrix(xp);
      auto const       svd = SVD<Cx>(patch.transpose());
      // Soft-threhold svals
      Eigen::VectorXf const s = (svd.S.abs() > realλ).select(svd.S * (svd.S.abs() - realλ) / svd.S.abs(), 0.f);
      patch = (svd.U * s.asDiagonal() * svd.V.adjoint()).transpose();
      Cx4 yp = Tensorfy(patch, xp.dimensions());
      return yp;
    };
    Patches(patchSize, windowSize, shift, softLLR, x, z);
    Log::Debug("LLR α {} λ {} t {} |x| {} |z| {}", realα->scale, λ, realλ, Norm(x), Norm(z));
  } else {
    Log::Fail("C++ is stupid");
  }
}

} // namespace rl::Proxs
