#include "llr.hpp"

#include "../algo/decomp.hpp"
#include "../log/log.hpp"
#include "../patches.hpp"
#include "../tensors.hpp"

namespace rl::Proxs {

template <int D> LLR<D>::LLR(float const l, Index const p, Index const w, bool const doShift, Sz<D> const s)
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
  Index const N = Product(LastN<D - 3>(s));
  if (N < 2) { throw(Log::Failure("LLR", "Need more than 1 image to perform LLR")); }
  Index const B = Product(FirstN<3>(s)) / (M * N);
  λ *= (std::sqrt(M) + std::sqrt(N) + std::sqrt(std::log(B * std::min(M, N))));
  Log::Print("LLR", "λ {} Scaled λ {} Patch {} Window {} M {} N {} Shape {}", l, λ, patchSize, windowSize, M, N, shape);
}

template <int D> void LLR<D>::apply(float const α, Map xin) const
{
  CxNCMap<D>     x(xin.data(), shape);
  float const realλ = λ * α;
  auto        softLLR = [realλ](CxN<D> const &xp, CxN<D> &yp) {
    auto const    xm = CollapseToConstMatrix<CxN<D>, 3>(xp);
    auto          ym = CollapseToMatrix<CxN<D>, 3>(yp);
    SVD<Cx> const svd(xm);
    // Soft-threhold svals
    Eigen::VectorXf const s = (svd.S.abs() > realλ).select(svd.S * (svd.S.abs() - realλ) / svd.S.abs(), 0.f);
    ym = (svd.U * s.asDiagonal() * svd.V.adjoint());
  };
  CxN<D> z(shape);
  z.setZero();
  Patches<D - 3>(patchSize, windowSize, shift, softLLR, x, z);
  Log::Debug("LLR", "α {:4.3E} λ {:4.3E} t {:4.3E} |x| {:4.3E} |z| {:4.3E}", α, λ, realλ, Norm<true>(x), Norm<true>(z));
  x.device(Threads::TensorDevice()) = z;
}

template <int D> void LLR<D>::conj(float const, Map xin) const
{
  CxNCMap<D> x(xin.data(), shape);
  /* Amazingly, this doesn't depend on α. Maths is based. */
  auto projLLR = [λ = this->λ](CxN<D> const &xp, CxN<D> &yp) {
    auto const            xm = CollapseToConstMatrix<CxN<D>, 3>(xp);
    auto                  ym = CollapseToMatrix<CxN<D>, 3>(yp);
    SVD<Cx> const         svd(xm);
    Eigen::VectorXf const s = (svd.S.abs() > λ).select(λ * svd.S / svd.S.abs(), svd.S);
    ym = (svd.U * s.asDiagonal() * svd.V.adjoint());
  };
  CxN<D> z(shape);
  Patches<D - 3>(patchSize, windowSize, shift, projLLR, x, z);
  Log::Debug("LLR", "Conjugate λ {:4.3E} |x| {:4.3E} |z| {:4.3E}", λ, Norm<true>(x), Norm<true>(z));
  x.device(Threads::TensorDevice()) = z;
}

template struct LLR<5>;
template struct LLR<6>;

} // namespace rl::Proxs
