#include "sense.h"

#include "fft3n.h"
#include "gridder.h"
#include "io_hd5.h"
#include "sdc.h"
#include "tensorOps.h"
#include "threads.h"

float const sense_res = 12.f;

Cx4 Direct(Gridder const &gridder, Cx3 const &data, Log &log)
{
  // Grid at low res & accumulate combined image
  Cx4 grid = gridder.newGrid();
  R3 rss(gridder.gridDims());
  FFT3N fftN(grid, log);
  grid.setZero();
  rss.setZero();
  gridder.toCartesian(data, grid);
  fftN.reverse();
  rss.device(Threads::GlobalDevice()) = (grid * grid.conjugate()).real().sum(Sz1{0}).sqrt();
  log.info("Normalizing channel images");
  grid.device(Threads::GlobalDevice()) = grid / TileToMatch(rss, grid.dimensions()).cast<Cx>();
  log.info("Finished SENSE maps");
  return grid;
}

Cx4 SENSE(
    std::string const &method,
    Trajectory const &traj,
    Gridder const &gridder,
    Cx3 const &data,
    Log &log)
{
  if (method == "direct") {
    log.info("Creating SENSE maps from main image data");
    Cx3 lo_data = data;
    auto const lo_traj = traj.trim(sense_res, lo_data);
    Gridder lo_gridder(lo_traj, gridder.oversample(), gridder.kernel(), false, log);
    SDC::Load("pipe", lo_traj, lo_gridder, log);
    return Direct(lo_gridder, lo_data, log);
  } else {
    log.info("Loading SENSE data from {}", method);
    HD5::Reader reader(method, log);
    Trajectory const cal_traj = reader.readTrajectory();
    Gridder cal_gridder(cal_traj, gridder.oversample(), gridder.kernel(), false, log);
    SDC::Load("pipe", cal_traj, cal_gridder, log);
    Cx3 cal_data = cal_traj.info().noncartesianVolume();
    reader.readNoncartesian(0, cal_data);
    return Direct(cal_gridder, cal_data, log);
  }
}