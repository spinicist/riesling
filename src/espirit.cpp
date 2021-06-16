#include "espirit.h"

#include "cropper.h"
#include "decomp.h"
#include "fft_plan.hpp"
#include "gridder.h"
#include "hankel.h"
#include "io_nifti.h"
#include "padder.h"
#include "tensorOps.h"
#include "threads.h"
#include "vc.h"

Cx4 ESPIRIT(
    Gridder const &hi_gridder,
    Gridder const &lo_gridder,
    Cx3 const &data,
    long const kRad,
    long const calRad,
    Log &log)
{
  log.info(FMT_STRING("ESPIRIT Calibration Radius {} Kernel Radius {}"), calRad, kRad);

  log.info(FMT_STRING("Calculating k-space kernels"));
  Cx4 lo_grid = lo_gridder.newGrid();
  FFT::ThreeDMulti lo_fft(lo_grid, log);
  lo_gridder.toCartesian(data, lo_grid);
  Cx5 const all_kernels = ToKernels(lo_grid, kRad, calRad, lo_gridder.info().read_gap, log);
  Cx5 const mini_kernels = LowRankKernels(all_kernels, 0.015, log);
  long const retain = mini_kernels.dimension(4);

  log.info(FMT_STRING("Transform to image kernels"));
  Cx4 hi_grid = hi_gridder.newGrid(); // Maps will end up here
  Cx4 hi_vals = hi_gridder.newGrid();
  Cx4 mix_grid(
      lo_grid.dimension(0), lo_grid.dimension(1), lo_grid.dimension(2), hi_grid.dimension(3));
  FFT::Plan<4, 1> mix_fft(mix_grid, log);

  // These will already have the last dimension upsampled
  Cx5 mix_kernels(
      lo_grid.dimension(0),
      lo_grid.dimension(1),
      lo_grid.dimension(2),
      hi_grid.dimension(3),
      retain);
  mix_kernels.setZero();
  Cropper const lo_mix(
      Sz3{mix_grid.dimension(1), mix_grid.dimension(2), mix_grid.dimension(3)},
      Sz3{mini_kernels.dimension(1), mini_kernels.dimension(2), mini_kernels.dimension(3)},
      log);
  float const scale =
      (1.f /
       sqrt(mini_kernels.dimension(1) * mini_kernels.dimension(2) * mini_kernels.dimension(3))) /
      lo_fft.scale();
  for (long kk = 0; kk < retain; kk++) {
    mix_grid.setZero();
    lo_mix.crop4(mix_grid) = mini_kernels.chip(kk, 4) * mini_kernels.chip(kk, 4).constant(scale);
    mix_fft.reverse(mix_grid);
    mix_kernels.chip(kk, 4) = mix_grid;
    log.progress(kk, 0, retain);
  }

  log.info(FMT_STRING("Image space Eigenanalysis"));
  // Do this slice-by-slice
  auto slice_task = [&hi_grid, &hi_vals, &mix_kernels, &log](long const lo_z, long const hi_z) {
    for (long zz = lo_z; zz < hi_z; zz++) {
      Cx4 hi_kernels(
          hi_grid.dimension(0),
          hi_grid.dimension(1),
          hi_grid.dimension(2),
          mix_kernels.dimension(4));
      Cx3 hi_slice(hi_grid.dimension(0), hi_grid.dimension(1), hi_grid.dimension(2));
      Log nullLog;
      FFT::Plan<3, 2> hi_slice_fft(hi_slice, nullLog, 1);

      // Now do a lot of FFTs
      for (long kk = 0; kk < mix_kernels.dimension(4); kk++) {
        Cx3 const mix_kernel = mix_kernels.chip(kk, 4).chip(zz, 3);
        hi_slice.setZero();
        CropLast2(hi_slice, mix_kernel.dimensions()) = mix_kernel;
        hi_slice_fft.reverse(hi_slice);
        hi_kernels.chip(kk, 3) = hi_slice;
      }

      // Now voxel-wise covariance
      for (long yy = 0; yy < hi_slice.dimension(2); yy++) {
        for (long xx = 0; xx < hi_slice.dimension(1); xx++) {
          Cx2 const samples = hi_kernels.chip(yy, 2).chip(xx, 1);
          Cx2 vecs(samples.dimension(0), samples.dimension(0));
          R1 vals(samples.dimension(0));
          PCA(samples, vecs, vals, log);
          Cx1 const vec0 = vecs.chip(0, 1);
          float const phase = std::arg(vec0(0));
          hi_grid.chip(zz, 3).chip(yy, 2).chip(xx, 1) =
              (vec0 * std::polar(1.f, -phase)).conjugate();
          hi_vals.chip(zz, 3).chip(yy, 2).chip(xx, 1) = vals.cast<Cx>();
        }
      }
      log.progress(zz, lo_z, hi_z);
    }
  };
  Threads::RangeFor(slice_task, mix_kernels.dimension(3));

  log.info("Finished ESPIRIT");
  log.image(hi_vals, "espirit-val.nii");
  log.image(hi_grid, "espirit-vec.nii");
  return hi_grid;
}