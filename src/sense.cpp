#include "sense.h"

#include "cropper.h"
#include "espirit.h"
#include "fft3n.h"
#include "filter.h"
#include "gridder.h"
#include "io_hd5.h"
#include "tensorOps.h"
#include "threads.h"
#include "vc.h"

float const sense_res = 8.f;

Cx4 Direct(Gridder const &gridder, Cx3 const &data, Log &log)
{
  // Grid at low res & accumulate combined image
  Cx4 grid = gridder.newGrid();
  R3 rss(gridder.gridDims());
  FFT3N fftN(grid, log);
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
    Cx4 sense = Direct(lo_gridder, lo_data, log);
    return sense;
  } else if (method == "espirit") {
    Cx3 lo_data = data;
    auto const lo_traj = traj.trim(sense_res, lo_data, true);
    log.image(
        Cx3(lo_data.slice(Sz3{0, 0, 0}, Sz3{lo_data.dimension(0), lo_data.dimension(1), 256})),
        "sense-lo-data.nii");
    Gridder lo_gridder(lo_traj, gridder.oversample(), gridder.kernel(), false, log);
    SDC::Load("pipe", lo_traj, lo_gridder, log);
    // Set this up for upsampling
    Cx4 lores = lo_gridder.newGrid();
    FFT3N lo_fft(lores, log);
    Cx4 hires = gridder.newGrid();
    FFT3N hi_fft(hires, log);
    Cropper const cropper(
        Dims3{hires.dimension(1), hires.dimension(2), hires.dimension(3)},
        Dims3{lores.dimension(1), lores.dimension(2), lores.dimension(3)},
        log);
    long const kRad = 4;
    long const calRad = kRad + 1 + (lo_gridder.info().spokes_lo ? 0 : lo_gridder.info().read_gap);
    lores = ESPIRIT(lo_gridder, lo_data, kRad, calRad, log);
    log.info(FMT_STRING("Upsample maps"));
    lo_fft.forward(lores);
    log.image(lores, "espirit-lores-ks.nii");
    KSTukey(0.5f, 1.f, 0.f, lores, log);
    log.image(lores, "espirit-lores-filtered.nii");
    hires.setZero();
    cropper.crop4(hires) = lores * lores.constant(lo_fft.scale() / hi_fft.scale());
    log.image(hires, "espirit-hires-ks.nii");
    hi_fft.reverse(hires);
    log.image(hires, "espirit-hires.nii");
    return hires;
  } else {
    log.info("Loading SENSE data from {}", method);
    HD5::Reader reader(method, log);
    Trajectory const cal_traj = reader.readTrajectory();
    Gridder cal_gridder(cal_traj, gridder.oversample(), gridder.kernel(), false, log);
    if ((cal_gridder.info().matrix != gridder.info().matrix).any()) {
      log.fail("Calibration data has incompatible matrix size");
    }
    SDC::Load("pipe", cal_traj, cal_gridder, log);
    Cx3 cal_data = cal_traj.info().noncartesianVolume();
    reader.readNoncartesian(0, cal_data);
    return Direct(cal_gridder, cal_data, log);
  }
}