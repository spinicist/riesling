#include "espirit.h"

#include "cropper.h"
#include "decomp.h"
#include "fft3n.h"
#include "gridder.h"
#include "hankel.h"
#include "io_nifti.h"
#include "padder.h"
#include "tensorOps.h"
#include "threads.h"

Cx4 ESPIRIT(Gridder const &gridder, Cx3 const &data, long const kRad, long const calRad, Log &log)
{
  log.info(FMT_STRING("ESPIRIT Calibration Radius {} Kernel Radius {}"), calRad, kRad);
  Cx4 lo_grid = gridder.newGrid();
  FFT3N lo_fft(lo_grid, log);
  lo_grid.setZero();
  gridder.toCartesian(data, lo_grid);
  log.image(lo_grid, "espirit-lo-grid.nii");
  lo_fft.reverse();
  log.image(lo_grid, "espirit-lo-img.nii");
  lo_fft.forward();
  log.info(FMT_STRING("Calculating k-space kernels"));
  Cx5 const all_mini_kernels = ToKernels(lo_grid, kRad, calRad, gridder.info().read_gap, log);
  log.image(Cx4(all_mini_kernels.chip(0, 4)), "espirit-mini-kernel0-ks.nii");
  log.image(
      Cx4(all_mini_kernels.chip(all_mini_kernels.dimension(4) / 2, 4)),
      "espirit-mini-kernel1-ks.nii");
  log.image(
      Cx4(all_mini_kernels.chip(all_mini_kernels.dimension(4) - 1, 4)),
      "espirit-mini-kernel2-ks.nii");
  Cx5 const mini_kernels = LowRankKernels(all_mini_kernels, 0.05, log);
  long const retain = mini_kernels.dimension(4);

  log.info(FMT_STRING("Transform to image kernels"));
  Cx5 lo_kernels(
      lo_grid.dimension(0),
      lo_grid.dimension(1),
      lo_grid.dimension(2),
      lo_grid.dimension(3),
      retain);
  lo_kernels.setZero();
  Cropper const lo_mini(
      Dims3{lo_grid.dimension(1), lo_grid.dimension(2), lo_grid.dimension(3)},
      Dims3{mini_kernels.dimension(1), mini_kernels.dimension(2), mini_kernels.dimension(3)},
      log);
  float const scale =
      (1.f /
       sqrt(mini_kernels.dimension(1) * mini_kernels.dimension(2) * mini_kernels.dimension(3))) /
      lo_fft.scale();
  for (long kk = 0; kk < retain; kk++) {
    lo_grid.setZero();
    lo_mini.crop4(lo_grid) = mini_kernels.chip(kk, 4) * mini_kernels.chip(kk, 4).constant(scale);
    if (kk == 0) {
      log.image(lo_grid, "espirit-lo-kernel0-ks.nii");
    }
    lo_fft.reverse(lo_grid);
    lo_kernels.chip(kk, 4) = lo_grid;
    log.progress(kk, 0, retain);
  }
  log.image(Cx4(lo_kernels.chip(0, 4)), "espirit-lo-kernel0-img.nii");
  log.image(Cx4(lo_kernels.chip(retain / 2, 4)), "espirit-lo-kernel1-img.nii");
  log.image(Cx4(lo_kernels.chip(retain - 1, 4)), "espirit-lo-kernel2-img.nii");
  log.info(FMT_STRING("Image space Eigenanalysis"));
  Cx4 vec = gridder.newGrid();
  Cx4 val = gridder.newGrid();
  auto cov_task = [&lo_kernels, &vec, &val, &log](long const lo_z, long const hi_z) {
    for (long zz = lo_z; zz < hi_z; zz++) {
      for (long yy = 0; yy < lo_kernels.dimension(2); yy++) {
        for (long xx = 0; xx < lo_kernels.dimension(1); xx++) {
          Cx2 const samples = lo_kernels.chip(zz, 3).chip(yy, 2).chip(xx, 1);
          Cx2 const vox_cov = Covariance(samples);
          Cx2 vecs(vox_cov.dimensions());
          R1 vals(vox_cov.dimension(0));
          PCA(vox_cov, vecs, vals, log);
          Cx1 const vec0 = vecs.chip(vecs.dimension(1) - 1, 1);
          float const phase = std::arg(vec0(0));
          vec.chip(zz, 3).chip(yy, 2).chip(xx, 1) = (vec0 * std::polar(1.f, -phase)).conjugate();
          val.chip(zz, 3).chip(yy, 2).chip(xx, 1) = vals.cast<Cx>();
        }
      }
      log.progress(zz, lo_z, hi_z);
    }
  };
  Threads::RangeFor(cov_task, lo_kernels.dimension(3));
  log.info("Finished ESPIRIT");
  // log.image(cov, "espirit-cov.nii");
  log.image(val, "espirit-val.nii");
  log.image(vec, "espirit-vec.nii");
  return vec;
}