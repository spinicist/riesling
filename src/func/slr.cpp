#include "slr.hpp"

#include "algo/decomp.hpp"
#include "cropper.h"
#include "log.hpp"
#include "tensorOps.hpp"
#include "threads.hpp"

namespace rl {
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
  Re3 count(LastN<3>(grid.dimensions()));
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
  grid /= count.reshape(AddFront(count.dimensions(), 1, 1)).broadcast(Sz5{nC, nF, 1, 1, 1}).cast<Cx>();
}

SLR::SLR(FFTOp<5> const &f, Index const k)
  : Prox<Cx5>()
  , fft{f}
  , kSz{k}
{
}

auto SLR::operator()(float const thresh, Cx5 const &channels) const -> Cx5
{
  Index const nC = channels.dimension(0); // Include frames here
  if (kSz < 3) {
    Log::Fail(FMT_STRING("SLR kernel size less than 3 not supported"));
  }
  if (thresh < 0.f || thresh >= nC) {
    Log::Fail(FMT_STRING("SLR window threshold {} out of range {}-{}"), thresh, 0.f, nC);
  }
  Log::Print(FMT_STRING("SLR regularization kernel size {} window-normalized thresh {}"), kSz, thresh);
  Cx5 grid = fft.forward(channels);
  // Now crop out corners which will have zeros
  Index const w = std::floor(grid.dimension(2) / sqrt(3));
  Log::Print(FMT_STRING("Extract width {}"), w);

  Cx5 cropped = CropLast3(grid, Sz3{w, w, w});
  Cx6 kernels = ToKernels(cropped, kSz);
  Log::Tensor(grid, "zin-slr-b4-grid");
  Log::Tensor(cropped, "zin-slr-b4-cropped");
  Log::Tensor(kernels, "zin-slr-b4-kernels");
  auto kMat = CollapseToMatrix<Cx6, 5>(kernels);
  if (kMat.rows() > kMat.cols()) {
    Log::Fail(FMT_STRING("Insufficient kernels for SVD {}x{}"), kMat.rows(), kMat.cols());
  }
  auto const svd = SVD<Cx>(kMat, true, true);
  Index const nK = kernels.dimension(1) * kernels.dimension(2) * kernels.dimension(3) * kernels.dimension(4);
  Index const nZero = (nC - thresh) * nK; // Window-Normalized
  Log::Print(FMT_STRING("Zeroing {} values check {} nK {}"), nZero, (nC - thresh), nK);
  auto lrVals = svd.vals;
  lrVals.tail(nZero).setZero();
  kMat = (svd.U * lrVals.matrix().asDiagonal() * svd.V.adjoint()).transpose();
  FromKernels(kernels, cropped);
  CropLast3(grid, Sz3{w, w, w}) = cropped;
  Log::Tensor(cropped, "zin-slr-after-cropped");
  Log::Tensor(grid, "zin-slr-after-grid");
  Log::Tensor(kernels, "zin-slr-after-kernels");
  Cx5 outChannels = fft.adjoint(grid);
  Log::Tensor(channels, "zin-slr-b4-channels");
  Log::Tensor(outChannels, "zin-slr-after-channels");
  return outChannels;
}
} // namespace rl
