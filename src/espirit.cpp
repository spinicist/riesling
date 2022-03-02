#include "espirit.h"

#include "cropper.h"
#include "decomp.h"
#include "fft/fft.hpp"
#include "hankel.h"
#include "op/grids.h"
#include "tensorOps.h"
#include "threads.h"
#include "vc.h"

Cx4 ESPIRIT(
  GridBase *gridder,
  Cx3 const &data,
  Index const kRad,
  Index const calRad,
  Index const gap,
  float const thresh)
{
  Log::Print(FMT_STRING("ESPIRIT Calibration Radius {} Kernel Radius {}"), calRad, kRad);

  Log::Print(FMT_STRING("Calculating k-space kernels"));
  Cx4 grid = gridder->Adj(data).chip<1>(0);

  Cx5 const all_kernels = ToKernels(grid, kRad, calRad, gap);
  Cx5 const mini_kernels = LowRankKernels(all_kernels, thresh);
  Index const retain = mini_kernels.dimension(4);

  Log::Print(FMT_STRING("Upsample last dimension"));
  Cx4 mix_grid(
    mini_kernels.dimension(0),
    mini_kernels.dimension(1),
    mini_kernels.dimension(2),
    grid.dimension(3));
  auto const mix_fft = FFT::Make<4, 1>(mix_grid.dimensions());
  Cx5 mix_kernels(
    mini_kernels.dimension(0),
    mini_kernels.dimension(1),
    mini_kernels.dimension(2),
    grid.dimension(3),
    retain);
  mix_kernels.setZero();
  Cropper const lo_mix(
    Sz3{mix_grid.dimension(1), mix_grid.dimension(2), mix_grid.dimension(3)},
    Sz3{mini_kernels.dimension(1), mini_kernels.dimension(2), mini_kernels.dimension(3)});
  float const scale =
    (1.f / sqrt(mini_kernels.dimension(1) * mini_kernels.dimension(2) * mini_kernels.dimension(3)));
  for (Index kk = 0; kk < retain; kk++) {
    mix_grid.setZero();
    lo_mix.crop4(mix_grid) = mini_kernels.chip<4>(kk) * mini_kernels.chip<4>(kk).constant(scale);
    mix_fft->reverse(mix_grid);
    mix_kernels.chip<4>(kk) = mix_grid;
    Log::Progress(kk, 0, retain);
  }

  Log::Print(FMT_STRING("Image space Eigenanalysis"));
  // Do this slice-by-slice
  R3 valsImage(gridder->mapping().cartDims);
  auto const hiSz = FirstN<3>(grid.dimensions());
  auto const hiFFT = FFT::Make<3, 2>(hiSz, 1);
  auto slice_task =
    [&grid, &valsImage, &mix_kernels, &hiSz, &hiFFT](Index const lo_z, Index const hi_z) {
      for (Index zz = lo_z; zz < hi_z; zz++) {
        Cx4 hi_kernels(
          grid.dimension(0), grid.dimension(1), grid.dimension(2), mix_kernels.dimension(4));
        Cx3 hi_slice(hiSz);

        // Now do a lot of FFTs
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
            Cx2 vecs(samples.dimension(0), samples.dimension(0));
            R1 vals(samples.dimension(0));
            PCA(samples, vecs, vals);
            Cx1 const vec0 = vecs.chip<1>(0);
            float const phase = std::arg(vec0(0));
            grid.chip<3>(zz).chip<2>(yy).chip<1>(xx) = (vec0 * std::polar(1.f, -phase)).conjugate();
            valsImage(xx, yy, zz) = vals(0);
          }
        }
        Log::Progress(zz, lo_z, hi_z);
      }
    };
  Threads::RangeFor(slice_task, mix_kernels.dimension(3));

  Log::Print(FMT_STRING("Finished ESPIRIT"));
  Log::Image(valsImage, "espirit-val.nii");
  return grid;
}