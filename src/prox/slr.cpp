#include "slr.hpp"

#include "algo/decomp.hpp"
#include "cropper.h"
#include "log.hpp"
#include "tensorOps.hpp"
#include "threads.hpp"

namespace rl::Proxs {

Cx6 ToKernels(Eigen::TensorMap<Cx5> const &grid, Index const kW)
{
  Index const nC = grid.dimension(0);
  Index const nF = grid.dimension(1);
  Index const nKx = grid.dimension(2) - kW + 1;
  Index const nKy = grid.dimension(3) - kW + 1;
  Index const nKz = grid.dimension(4) - kW + 1;
  Index const nK = nKx * nKy * nKz;
  if (nK < 1) { Log::Fail("No kernels to Hankelfy"); }
  Log::Print<Log::Level::Debug>("Hankelfying {} kernels", nK);
  Cx6   kernels(nC, nF, kW, kW, kW, nK);
  Index ik = 0;
  for (Index iz = 0; iz < nKx; iz++) {
    for (Index iy = 0; iy < nKy; iy++) {
      for (Index ix = 0; ix < nKz; ix++) {
        Sz5 st{0, 0, ix, iy, iz};
        Sz5 sz{nC, nF, kW, kW, kW};
        kernels.chip<5>(ik++) = grid.slice(st, sz);
      }
    }
  }
  assert(ik == nK);
  return kernels;
}

void FromKernels(Cx6 const &kernels, Eigen::TensorMap<Cx5> &grid)
{
  Index const kX = kernels.dimension(2);
  Index const kY = kernels.dimension(3);
  Index const kZ = kernels.dimension(4);
  Index const nK = kernels.dimension(5);
  Index const nC = grid.dimension(0);
  Index const nF = grid.dimension(1);
  Index const nX = grid.dimension(2) - kX + 1;
  Index const nY = grid.dimension(3) - kY + 1;
  Index const nZ = grid.dimension(4) - kZ + 1;
  Re3         count(LastN<3>(grid.dimensions()));
  count.setZero();
  grid.setZero();
  Index ik = 0;
  Log::Print<Log::Level::Debug>("Unhankelfying {} kernels", nK);
  for (Index iz = 0; iz < nZ; iz++) {
    for (Index iy = 0; iy < nY; iy++) {
      for (Index ix = 0; ix < nX; ix++) {
        grid.slice(Sz5{0, 0, ix, iy, iz}, Sz5{nC, nF, kX, kY, kZ}) += kernels.chip<5>(ik++);
        count.slice(Sz3{ix, iy, iz}, Sz3{kX, kY, kZ}) += count.slice(Sz3{ix, iy, iz}, Sz3{kX, kY, kZ}).constant(1.f);
      }
    }
  }
  assert(ik == nK);
  grid /= count.reshape(AddFront(count.dimensions(), 1, 1)).broadcast(Sz5{nC, nF, 1, 1, 1}).cast<Cx>();
}

SLR::SLR(float const l, Index const k, Sz5 const sh)
  : Prox<Cx>(Product(sh))
  , λ{l}
  , kSz{k}
  , shape{sh}
  , fft{FFT::Make<5, 3>(shape)}
{
  if (kSz < 3) { Log::Fail("SLR kernel size less than 3 not supported"); }
  Log::Print("Structured Low-Rank λ {} Kernel-size {} Shape {}", λ, kSz, shape);
}

void SLR::apply(float const α, CMap const &xin, Map &zin) const
{
  Eigen::TensorMap<Cx5 const> x(xin.data(), shape);
  float const                 thresh = λ * α;
  Eigen::TensorMap<Cx5>       z(zin.data(), shape);
  z = x;
  fft->forward(z);
  Cx6  kernels = ToKernels(z, kSz);
  auto kMat = CollapseToMatrix<Cx6, 5>(kernels);
  if (kMat.rows() > kMat.cols()) { Log::Fail("Insufficient kernels for SVD {}x{}", kMat.rows(), kMat.cols()); }
  auto const            svd = SVD<Cx>(kMat.transpose());
  Eigen::VectorXf const s = (svd.S.array() > thresh).select(svd.S, 0.f);
  kMat = (svd.U * s.asDiagonal() * svd.V.adjoint()).transpose();
  FromKernels(kernels, z);
  fft->reverse(z);
  Log::Print<Log::Level::High>("SLR α {} λ {} t {} |x| {} |z| {} s {}", α, λ, thresh, Norm(x), Norm(z), s.head(5).transpose());
}

} // namespace rl::Proxs
