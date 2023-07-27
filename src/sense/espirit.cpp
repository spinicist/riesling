#include "sense/espirit.hpp"

#include "algo/decomp.hpp"
#include "algo/stats.hpp"
#include "cropper.h"
#include "fft/fft.hpp"
#include "tensorOps.hpp"
#include "threads.hpp"

namespace rl {
namespace SENSE {

Cx5 ToKernels(Cx4 const &grid, Index const kRad, Index const calRad, Index const gapRad)
{
  Sz4 const   fullShape = grid.dimensions();
  Index const nchan = fullShape[0];
  Sz3 const   halfShape = Div(LastN<3>(fullShape), 2);

  Index const calW = (calRad * 2) - 1;
  Index const kW = (kRad * 2) - 1;
  Index const gapPlusKW = ((gapRad + kRad) * 2) - 1;
  Index const nSkip = gapRad ? gapPlusKW * gapPlusKW * gapPlusKW : 0;
  Index const nk = calW * calW * calW - nSkip;
  if (nk < 1) { Log::Fail("No kernels to Hankelfy"); }
  Cx5 kernels(nchan, kW, kW, kW, nk);

  Index       k = 0;
  Index const gapSt = (calRad - 1) - (gapRad - 1) - kRad;
  Index const gapEnd = (calRad - 1) + gapRad + kRad;

  Sz3 const st = Add(halfShape, -(calRad - 1) - (kRad - 1));
  for (Index ii = 0; ii < 3; ii++) {
    if (st[ii] < 0) {
      Log::Fail("Grid size {} not large enough for calibration radius {} + kernel radius {}", grid.dimension(1), calRad, kRad);
    }
  }

  Log::Print("Hankel calibration rad {} kernel rad {} gap {}, {} kernels", calRad, kRad, gapRad, nk);
  for (Index iz = 0; iz < calW; iz++) {
    Index const st_z = st[2] + iz;
    for (Index iy = 0; iy < calW; iy++) {
      Index const st_y = st[1] + iy;
      for (Index ix = 0; ix < calW; ix++) {
        if (gapRad && (ix >= gapSt && ix < gapEnd) && (iy >= gapSt && iy < gapEnd) && (iz >= gapSt && iz < gapEnd)) {
          continue;
        }
        Index const st_x = st[0] + ix;
        Sz4         sst{0, st_x, st_y, st_z};
        Sz4         ssz{nchan, kW, kW, kW};
        kernels.chip<4>(k) = grid.slice(sst, ssz);
        k++;
      }
    }
  }
  assert(k == nk);
  return kernels;
}

Cx5 LowRankKernels(Cx5 const &mIn, float const thresh)
{
  auto const  m = CollapseToMatrix<Cx5, 4>(mIn);
  auto const  svd = SVD<Cx>(m.transpose());
  Index const nRetain = (svd.S.array() > (svd.S.sum() * thresh)).count();
  Log::Print("Retaining {} kernels", nRetain);
  Cx5 out(mIn.dimension(0), mIn.dimension(1), mIn.dimension(2), mIn.dimension(3), nRetain);
  CollapseToMatrix<Cx5, 4>(out) = svd.V.leftCols(nRetain).conjugate();
  return out;
}

Cx4 ESPIRIT(Cx4 const &grid, Sz3 const outShape, Index const kRad, Index const calRad, Index const gap, float const thresh)
{
  Log::Print("ESPIRIT Calibration Radius {} Kernel Radius {}", calRad, kRad);

  Log::Print("Calculating k-space kernels");
  Cx5 const   all_kernels = ToKernels(grid, kRad, calRad, gap);
  Cx5 const   mini_kernels = LowRankKernels(all_kernels, thresh);
  Index const retain = mini_kernels.dimension(4);

  Log::Print("Upsample last dimension");
  Sz3 const   inShape = LastN<3>(grid.dimensions());
  Index const nC = grid.dimension(0);
  Cx4         mix_grid(mini_kernels.dimension(0), mini_kernels.dimension(1), mini_kernels.dimension(2), inShape[2]);
  auto const  mix_fft = FFT::Make<4, 1>(mix_grid.dimensions());
  Cx5         mix_kernels(mini_kernels.dimension(0), mini_kernels.dimension(1), mini_kernels.dimension(2), inShape[2], retain);
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

  Log::Print("Image space Eigenanalysis");
  // Do this slice-by-slice
  Re3        valsImage(inShape);
  Cx4        vecsImage(AddFront(inShape, nC));
  auto const hiSz = FirstN<3>(vecsImage.dimensions());
  auto const hiFFT = FFT::Make<3, 2>(hiSz, 1);
  auto       slice_task = [&vecsImage, &valsImage, &mix_kernels, &hiSz, &hiFFT](Index const zz) {
    // Now do a lot of FFTs
    Cx4 hi_kernels(vecsImage.dimension(0), vecsImage.dimension(1), vecsImage.dimension(2), mix_kernels.dimension(4));
    Cx3 hi_slice(hiSz);
    for (Index kk = 0; kk < mix_kernels.dimension(4); kk++) {
      Cx3 const mix_kernel = mix_kernels.chip<4>(kk).chip<3>(zz);
      hi_slice.setZero();
      Crop(hi_slice, mix_kernel.dimensions()) = mix_kernel;
      hiFFT->reverse(hi_slice);
      hi_kernels.chip<3>(kk) = hi_slice;
    }

    // Now voxel-wise covariance
    for (Index yy = 0; yy < hi_slice.dimension(2); yy++) {
      for (Index xx = 0; xx < hi_slice.dimension(1); xx++) {
        Cx2 const   samples = hi_kernels.chip<2>(yy).template chip<1>(xx);
        auto const cov = Covariance(CollapseToConstMatrix(samples));
        Eig<Cx> const eig(cov);
        float const phase = std::arg(eig.P(0, 0));
        for (Index ic = 0; ic < samples.dimension(0); ic++) {
          vecsImage(ic, xx, yy, zz) = std::conj(eig.P(ic, 0) * std::polar(1.f, -phase));
        }
        valsImage(xx, yy, zz) = eig.V[0];
      }
    }
  };
  Threads::For(slice_task, mix_kernels.dimension(3), "Covariance");

  Log::Print("Finished ESPIRIT");
  Cx4 const cropped = Crop(vecsImage, AddFront(outShape, nC));
  return cropped;
}

} // namespace SENSE
} // namespace rl
