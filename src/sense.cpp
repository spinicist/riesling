#include "sense.h"

#include "cropper.h"
#include "espirit.h"
#include "fft_plan.h"
#include "filter.h"
#include "gridder.h"
#include "io_hd5.h"
#include "tensorOps.h"
#include "threads.h"
#include "vc.h"

float const sense_res = 8.f;

Cx4 DirectSENSE(
    Trajectory const &traj,
    float const os,
    Kernel *kernel,
    float const fov,
    Cx3 const &data,
    float const lambda,
    Log &log)
{
  Cx3 lo_ks = data;
  auto const lo_traj = traj.trim(8.f, lo_ks);
  Gridder gridder(lo_traj, os, kernel, false, log);
  SDC::Load("pipe", lo_traj, gridder, log);

  Cx4 grid = gridder.newMultichannel(data.dimension(0));
  R3 rss(gridder.gridDims());
  FFT::ThreeDMulti fftN(grid, log);
  grid.setZero();
  rss.setZero();
  gridder.toCartesian(data, grid);

  float const end_rad = gridder.info().voxel_size.minCoeff() / sense_res;
  float const start_rad = 0.5 * end_rad;
  log.info(
      FMT_STRING("SENSE res {} image res {} oversample {} filter {}-{}"),
      sense_res,
      gridder.info().voxel_size.minCoeff(),
      gridder.oversample(),
      start_rad,
      end_rad);
  KSTukey(start_rad, end_rad, 0.f, grid, log);
  fftN.reverse(grid);
  if (lambda > 0.f) {
    log.info(FMT_STRING("Regularization lambda {}"), lambda);
    rss.device(Threads::GlobalDevice()) =
        (grid * grid.conjugate()).real().sum(Sz1{0}).sqrt() + rss.constant(lambda);
  } else {
    rss.device(Threads::GlobalDevice()) = (grid * grid.conjugate()).real().sum(Sz1{0}).sqrt();
  }
  log.info("Normalizing channel images");
  grid.device(Threads::GlobalDevice()) = grid / TileToMatch(rss, grid.dimensions()).cast<Cx>();
  log.info("Finished SENSE maps");

  Cropper crop(traj.info(), gridder.gridDims(), fov, log);
  return crop.crop4(grid);
}

Cx4 LoadSENSE(std::string const &calFile, Sz4 const dims, Log &log)
{
  Cx4 sense(dims);
  HD5::Reader senseReader(calFile, log);
  senseReader.readSENSE(sense);
  return sense;
}
