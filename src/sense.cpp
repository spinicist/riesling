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
  gridder.toCartesian(lo_ks, grid);

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

Cx4 InterpSENSE(std::string const &file, Eigen::Array3l const dims, Log &log)
{
  HD5::Reader senseReader(file, log);
  Cx4 disk_sense = senseReader.readSENSE();
  log.info("Interpolating SENSE maps to dimensions {}", dims.transpose());
  FFT::ThreeDMulti fft1(disk_sense.dimensions(), log);
  fft1.forward(disk_sense);
  Sz3 size1{disk_sense.dimension(1), disk_sense.dimension(2), disk_sense.dimension(3)};
  Sz3 size2{dims[0], dims[1], dims[2]};
  Cx4 sense(disk_sense.dimension(0), dims[0], dims[1], dims[2]);
  FFT::ThreeDMulti fft2(sense, log);
  sense.setZero();
  if (size1[0] < size2[0]) {
    Crop4(sense, size1) = disk_sense;
  } else {
    sense = Crop4(disk_sense, size2);
  }
  fft2.reverse(sense);
  return sense;
}