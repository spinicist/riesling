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

Cx4 DirectSENSE(Gridder const &gridder, Cx3 const &data, float const lambda, Log &log)
{
  // Grid at low res & accumulate combined image
  Cx4 grid = gridder.newGrid();
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
  return grid;
}

Cx4 LoadSENSE(
    long const nChan,
    Cropper const &cropper,
    std::string const &calFile,
    long const calVolume,
    HD5::Reader &reader,
    Trajectory const &traj,
    float const os,
    Kernel *kernel,
    float const lambda,
    Cx3 &rad_ks,
    long &currentVol,
    Log &log)
{
  Cx4 sense(cropper.dims(nChan));
  long currentVolume;

  if (calFile.empty()) {
    currentVolume = calVolume;
    reader.readNoncartesian(calVolume, rad_ks);
    Cx3 lo_ks = rad_ks;
    auto const lo_traj = traj.trim(8.f, lo_ks);
    Gridder lo_gridder(lo_traj, os, kernel, false, log);
    SDC::Load("pipe", lo_traj, lo_gridder, log);
    sense = cropper.crop4(DirectSENSE(lo_gridder, rad_ks, lambda, log));
  } else {
    HD5::Reader senseReader(calFile, log);
    senseReader.readSENSE(sense);
    currentVolume = -1;
  }
  currentVol = currentVolume;
  return sense;
}