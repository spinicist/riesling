#include "sense.h"

#include "cropper.h"
#include "espirit.h"
#include "fft_plan.h"
#include "filter.h"
#include "io_hd5.h"
#include "op/grid.h"
#include "sdc.h"
#include "tensorOps.h"
#include "threads.h"
#include "vc.h"

float const sense_res = 8.f;

Cx4 DirectSENSE(
    Trajectory const &traj,
    float const os,
    bool const kb,
    float const fov,
    Cx3 const &data,
    float const lambda,
    Log &log)
{
  auto gridder = make_grid(traj, os, kb, false, log, 8.f, false);
  SDC::Choose("pipe", traj, gridder, log);

  Cx4 grid = gridder->newMultichannel(data.dimension(0));
  FFT::ThreeDMulti fftN(grid, log);
  gridder->Adj(data, grid);
  log.image(grid, "sense-grid.nii");
  float const end_rad = traj.info().voxel_size.minCoeff() / sense_res;
  float const start_rad = 0.5 * end_rad;
  log.info(FMT_STRING("SENSE res {} filter {}-{}"), sense_res, start_rad, end_rad);
  KSTukey(start_rad, end_rad, 0.f, grid, log);
  log.image(grid, "sense-filtered.nii");
  fftN.reverse(grid);
  log.image(grid, "sense-fft.nii");

  Cropper crop(traj.info(), gridder->gridDims(), fov, log);
  Cx4 channels = crop.crop4(grid);
  Cx3 rss = crop.newImage();
  rss.device(Threads::GlobalDevice()) = ConjugateSum(channels, channels).sqrt();
  if (lambda) {
    log.info(FMT_STRING("Regularization lambda {}"), lambda);
    rss.device(Threads::GlobalDevice()) = rss + rss.constant(lambda);
  }
  log.image(rss, "sense-rss.nii");
  log.image(channels, "sense-channels.nii");
  log.info("Normalizing channel images");
  channels.device(Threads::GlobalDevice()) = channels / TileToMatch(rss, channels.dimensions());
  log.info("Finished SENSE maps");
  return channels;
}

Cx4 LoadSENSE(std::string const &calFile, Log &log)
{
  HD5::Reader senseReader(calFile, log);
  return senseReader.readSENSE();
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