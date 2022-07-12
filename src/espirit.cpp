#include "espirit.h"

#include "algo/decomp.h"
#include "cropper.h"
#include "fft/fft.hpp"
#include "op/gridBase.hpp"
#include "tensorOps.h"
#include "threads.h"
#include "vc.h"

namespace rl {

Cx5 ToKernels(Cx4 const &grid, Index const kRad, Index const calRad, Index const gapRad)
{
  Index const nchan = grid.dimension(0);
  Index const gridHalf = grid.dimension(1) / 2;
  Index const calW = (calRad * 2) - 1;
  Index const kW = (kRad * 2) - 1;
  Index const gapPlusKW = ((gapRad + kRad) * 2) - 1;
  Index const nSkip = gapRad ? gapPlusKW * gapPlusKW * gapPlusKW : 0;
  Index const nk = calW * calW * calW - nSkip;
  if (nk < 1) {
    Log::Fail(FMT_STRING("No kernels to Hankelfy"));
  }
  Cx5 kernels(nchan, kW, kW, kW, nk);

  Index k = 0;
  Index s = 0;
  Index const gapSt = (calRad - 1) - (gapRad - 1) - kRad;
  Index const gapEnd = (calRad - 1) + gapRad + kRad;

  Index const st = gridHalf - (calRad - 1) - (kRad - 1);
  if (st < 0) {
    Log::Fail(
      FMT_STRING("Grid size {} not large enough for calibration radius {} + kernel radius {}"),
      grid.dimension(1),
      calRad,
      kRad);
  }

  Log::Print(FMT_STRING("Hankel calibration rad {} kernel rad {} gap {}, {} kernels"), calRad, kRad, gapRad, nk);
  for (Index iz = 0; iz < calW; iz++) {
    Index const st_z = st + iz;
    for (Index iy = 0; iy < calW; iy++) {
      Index const st_y = st + iy;
      for (Index ix = 0; ix < calW; ix++) {
        if (gapRad && (ix >= gapSt && ix < gapEnd) && (iy >= gapSt && iy < gapEnd) && (iz >= gapSt && iz < gapEnd)) {
          s++;
          continue;
        }
        Index const st_x = st + ix;
        Sz4 sst{0, st_x, st_y, st_z};
        Sz4 ssz{nchan, kW, kW, kW};
        kernels.chip<4>(k) = grid.slice(sst, ssz);
        k++;
      }
    }
  }
  assert(s == nSkip);
  assert(k == nk);
  return kernels;
}

Cx5 LowRankKernels(Cx5 const &mIn, float const thresh)
{
  auto const m = CollapseToMatrix<Cx5, 4>(mIn);
  auto const svd = SVD<Cx>(m, true, true);
  Index const nRetain = (svd.vals > (svd.vals.sum() * thresh)).count();
  Log::Print(FMT_STRING("Retaining {} kernels"), nRetain);
  Cx5 out(mIn.dimension(0), mIn.dimension(1), mIn.dimension(2), mIn.dimension(3), nRetain);
  CollapseToMatrix<Cx5, 1>(out) = svd.V.leftCols(nRetain).conjugate();
  return out;
}

Cx4 ESPIRIT(
  GridBase<Cx> *gridder, Cx3 const &data, Index const kRad, Index const calRad, Index const gap, float const thresh)
{
  Log::Print(FMT_STRING("ESPIRIT Calibration Radius {} Kernel Radius {}"), calRad, kRad);

  Log::Print(FMT_STRING("Calculating k-space kernels"));
  Cx4 grid = gridder->Adj(data).chip<1>(0);

  Cx5 const all_kernels = ToKernels(grid, kRad, calRad, gap);
  Cx5 const mini_kernels = LowRankKernels(all_kernels, thresh);
  Index const retain = mini_kernels.dimension(4);

  Log::Print(FMT_STRING("Upsample last dimension"));
  Cx4 mix_grid(mini_kernels.dimension(0), mini_kernels.dimension(1), mini_kernels.dimension(2), grid.dimension(3));
  auto const mix_fft = FFT::Make<4, 1>(mix_grid.dimensions());
  Cx5 mix_kernels(
    mini_kernels.dimension(0), mini_kernels.dimension(1), mini_kernels.dimension(2), grid.dimension(3), retain);
  mix_kernels.setZero();
  Cropper const lo_mix(
    Sz3{mix_grid.dimension(1), mix_grid.dimension(2), mix_grid.dimension(3)},
    Sz3{mini_kernels.dimension(1), mini_kernels.dimension(2), mini_kernels.dimension(3)});
  float const scale = (1.f / sqrt(mini_kernels.dimension(1) * mini_kernels.dimension(2) * mini_kernels.dimension(3)));
  Log::StartProgress(retain, "FFT Kernels");
  for (Index kk = 0; kk < retain; kk++) {
    mix_grid.setZero();
    lo_mix.crop4(mix_grid) = mini_kernels.chip<4>(kk) * mini_kernels.chip<4>(kk).constant(scale);
    mix_fft->reverse(mix_grid);
    mix_kernels.chip<4>(kk) = mix_grid;
    Log::Tick();
  }
  Log::StopProgress();

  Log::Print(FMT_STRING("Image space Eigenanalysis"));
  // Do this slice-by-slice
  R3 valsImage(gridder->mapping().cartDims);
  auto const hiSz = FirstN<3>(grid.dimensions());
  auto const hiFFT = FFT::Make<3, 2>(hiSz, 1);
  auto slice_task = [&grid, &valsImage, &mix_kernels, &hiSz, &hiFFT](Index const zz) {
    // Now do a lot of FFTs
    Cx4 hi_kernels(grid.dimension(0), grid.dimension(1), grid.dimension(2), mix_kernels.dimension(4));
    Cx3 hi_slice(hiSz);
    for (Index kk = 0; kk < mix_kernels.dimension(4); kk++) {
      Cx3 const mix_kernel = mix_kernels.chip<4>(kk).chip<3>(zz);
      hi_slice.setZero();
      CropLast2(hi_slice, mix_kernel.dimensions()) = mix_kernel;
      hiFFT->reverse(hi_slice);
      hi_kernels.chip<3>(kk) = hi_slice;
    }

    // Now voxel-wise covariance
    for (Index yy = 0; yy < hi_slice.dimension(2); yy++) {
      for (Index xx = 0; xx < hi_slice.dimension(1); xx++) {
        Cx2 const samples = hi_kernels.chip<2>(yy).template chip<1>(xx);
        auto const pcs = PCA(CollapseToConstMatrix(samples), 1);
        float const phase = std::arg(pcs.vecs(0, 0));
        for (Index ic = 0; ic < samples.dimension(0); ic++) {
          grid(ic, xx, yy, zz) = std::conj(pcs.vecs(ic, 0) * std::polar(1.f, -phase));
        }
        valsImage(xx, yy, zz) = pcs.vals[0];
      }
    }
  };
  Threads::For(slice_task, mix_kernels.dimension(3), "Covariance");

  Log::Print(FMT_STRING("Finished ESPIRIT"));
  Log::Tensor(valsImage, "espirit-val");
  return grid;
}

} // namespace rl
