#include "slr.hpp"

#include "algo/decomp.hpp"
#include "cropper.hpp"
#include "log.hpp"
#include "tensors.hpp"
#include "threads.hpp"

namespace rl::Proxs {

template <int ND>
SLR<ND>::SLR(float const l, Sz<ND> const sh)
  : Prox<Cx>(Product(sh))
  , λ{l * std::sqrtf(Product(LastN<ND - 1>(sh)))}
  , shape{sh}
{
  Log::Print("Structured Low-Rank λ {} Scaled λ {} Shape {}", l, λ, shape);
}

template <int ND> void SLR<ND>::apply(float const α, CMap const &xin, Map &zin) const
{
  Eigen::TensorMap<CxN<ND> const> const x(xin.data(), shape);

  float const                           thresh = λ * α;
  Eigen::TensorMap<CxN<ND>>             z(zin.data(), shape);
  auto                                  xMat = CollapseToMatrix(x);
  auto                                  zMat = CollapseToMatrix(z);
  Log::Debug("Kernels {} as matrix {} {}", x.dimensions(), xMat.rows(), xMat.cols());
  auto const svd = SVD<Cx>(xMat.transpose());
  Log::Debug("U {} {} S {} V {} {}", svd.U.rows(), svd.U.cols(), svd.S.rows(), svd.V.rows(), svd.V.cols());
  Eigen::VectorXf const s = (svd.S.abs() > thresh).select(svd.S * (svd.S.abs() - thresh) / svd.S.abs(), 0.f);
  zMat = (svd.U * s.asDiagonal() * svd.V.adjoint()).transpose();
  Log::Debug("SLR α {} λ {} t {} |x| {} |z| {} s [{:1.2E} ... {:1.2E}]", α, λ, thresh, Norm(x), Norm(z),
             fmt::join(s.head(5), ", "), fmt::join(s.tail(5), ", "));
}

template struct SLR<6>;

} // namespace rl::Proxs
