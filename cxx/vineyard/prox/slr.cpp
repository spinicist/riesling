#include "slr.hpp"

#include "algo/decomp.hpp"
#include "log.hpp"
#include "tensors.hpp"
#include "threads.hpp"

namespace rl::Proxs {

template <int NK>
SLR<NK>::SLR(float const l, Sz5 const sh, Sz<NK> const dims, Sz<NK> const kW, bool const virt)
  : Prox<Cx>(Product(sh))
  , λ{l}
  , shape{sh}
  , F{shape}
  , H{shape, dims, kW, virt}
{
  λ *= (std::sqrt(H.rows()) + std::sqrt(H.cols()));
  Log::Print("Structured Low-Rank λ {} Scaled λ {} Shape {}", l, λ, shape);
}

template <int NK> void SLR<NK>::apply(float const α, CMap const &xin, Map &zin) const
{
  Eigen::TensorMap<Cx5 const> const x(xin.data(), shape);
  Eigen::TensorMap<Cx5>             z(zin.data(), shape);

  auto tmp = F.forward(x);
  auto k = H.forward(tmp);

  float const thresh = λ * α;
  auto        kMat = CollapseToMatrix(k);
  Log::Debug("Hankel {} as matrix {} {} norm {}", k.dimensions(), kMat.rows(), kMat.cols(), kMat.stableNorm());
  auto const svd = SVD<Cx>(kMat);
  Log::Debug("U {} {} S {} V {} {}", svd.U.rows(), svd.U.cols(), svd.S.rows(), svd.V.rows(), svd.V.cols());
  Eigen::VectorXf const s = (svd.S.abs() > thresh).select(svd.S * (svd.S.abs() - thresh) / svd.S.abs(), 0.f);
  kMat = (svd.U * s.asDiagonal() * svd.V.adjoint());
  tmp = H.adjoint(k);
  z = F.adjoint(tmp);
  Log::Print("SLR α {} λ {} t {} |x| {} |z| {} Retained {}/{}", α, λ, thresh, Norm(x), Norm(z), (svd.S.abs() > thresh).count(),
             svd.S.rows());
}

template struct SLR<1>;
template struct SLR<3>;

} // namespace rl::Proxs
