#include "sense.h"

#include "cropper.h"
#include "fft.h"
#include "gridder.h"
#include "log.h"
#include "threads.h"

Cx4 SENSE(
    Info const &info,
    R3 const &traj,
    float const os,
    bool const stack,
    bool const kb,
    Cx3 const &data,
    Log &log)
{
  // Grid and heavily smooth each coil image, accumulate combined image
  float const sense_res = 4.f;
  log.info("Creating SENSE maps.");
  Gridder gridder(info, traj, os, true, kb, stack, log, sense_res, false);
  gridder.setDCExponent(0.5);
  auto grid_sz = gridder.gridDims();
  Cx4 grid = gridder.newGrid();
  Cx3 rss(grid_sz);
  FFT3N fftN(grid, log);
  grid.setZero();
  rss.setZero();
  gridder.toCartesian(data, grid);
  fftN.reverse();
  rss.device(Threads::GlobalDevice()) = (grid * grid.conjugate()).sum(Sz1{0}).sqrt();
  log.info("Normalizing channel images");
  grid = grid / tile(rss, info.channels);
  log.info("Finished SENSE maps");
  return grid;
}
