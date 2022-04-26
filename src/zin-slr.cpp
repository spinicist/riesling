#include "zin-slr.hpp"

#include "algo/decomp.h"
#include "log.h"
#include "tensorOps.h"
#include "threads.h"

Cx6 ToKernels(Cx5 const &grid, Index const kW)
{
  Index const nC = grid.dimension(0);
  Index const nF = grid.dimension(1);
  Index const nKx = grid.dimension(2) - kW + 1;
  Index const nKy = grid.dimension(3) - kW + 1;
  Index const nKz = grid.dimension(4) - kW + 1;
  Index const nK = nKx * nKy * nKz;
  if (nK < 1) {
    Log::Fail(FMT_STRING("No kernels to Hankelfy"));
  }
  Log::Print(FMT_STRING("Hankelfying {} kernels"), nK);
  Log::Image(grid, "zin-slr-to-grid");
  Cx6 kernels(nC, nF, kW, kW, kW, nK);
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

void FromKernels(Cx6 const &kernels, Cx5 &grid)
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
  R3 count(LastN<3>(grid.dimensions()));
  count.setZero();
  grid.setZero();
  Index ik = 0;
  Log::Print(FMT_STRING("Unhankelfying {} kernels"), nK);
  for (Index iz = 0; iz < nZ; iz++) {
    for (Index iy = 0; iy < nY; iy++) {
      for (Index ix = 0; ix < nX; ix++) {
        grid.slice(Sz5{0, 0, ix, iy, iz}, Sz5{nC, nF, kX, kY, kZ}) += kernels.chip<5>(ik++);
        count.slice(Sz3{ix, iy, iz}, Sz3{kX, kY, kZ}) += count.slice(Sz3{ix, iy, iz}, Sz3{kX, kY, kZ}).constant(1.f);
      }
    }
  }
  assert(ik == nK);
  Log::Image(count, "zin-slr-from-count");
  grid /= count.reshape(AddFront(count.dimensions(), 1, 1)).broadcast(Sz5{nC, nF, 1, 1, 1}).cast<Cx>();
  Log::Image(grid, "zin-slr-from-grid");
}

Cx5 zinSLR(Cx5 const &channels, FFTOp<5> const &fft, Index const kSz, float const wnThresh)
{
  Log::Print(FMT_STRING("SLR regularization kernel size {} window-normalized thresh {}"), kSz, wnThresh);
  Cx5 kspaces = fft.A(channels);
  Cx6 kernels = ToKernels(kspaces, kSz);
  auto kMat = CollapseToMatrix<Cx6, 5>(kernels);
  auto const svd = SVD<Cx>(kMat, true, true);
  Index const nK = kernels.dimension(1) * kernels.dimension(2) * kernels.dimension(3) * kernels.dimension(4);
  Index const nC = kernels.dimension(0); // Include frames here
  Index const nZero = (nC - wnThresh) * nK; // Window-Normalized 
  Log::Print(FMT_STRING("Zeroing {} values check {} nK {}"), nZero, (nC - wnThresh), nK);
  auto lrVals = svd.vals;
  lrVals.tail(nZero).setZero();
  kMat = (svd.U * lrVals.matrix().asDiagonal() * svd.V.adjoint()).transpose();
  FromKernels(kernels, kspaces);
  Cx5 outChannels = fft.Adj(kspaces);
  Log::Image(outChannels, "zin-slr-channels");
  return outChannels;
}