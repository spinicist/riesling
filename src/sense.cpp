#include "sense.h"

#include "cropper.h"
#include "fft_plan.h"
#include "filter.h"
#include "io.h"
#include "tensorOps.h"
#include "threads.h"

Cx4 SelfCalibration(
  Info const &info,
  GridBase *gridder,
  float const fov,
  float const res,
  float const lambda,
  Cx3 const &data)
{
  Log::Debug(FMT_STRING("*** Self-Calibrated SENSE ***"));
  Sz5 const dims = gridder->inputDimensions();
  Cx4 grid(dims[0], dims[2], dims[3], dims[4]);
  FFT::Planned<4, 3> fftN(grid);
  grid = gridder->Adj(data).chip<1>(0); // Assume we want the first echo
  float const end_rad = info.voxel_size.minCoeff() / res;
  float const start_rad = 0.5 * end_rad;
  Log::Print(FMT_STRING("SENSE res {} filter {}-{}"), res, start_rad, end_rad);
  KSTukey(start_rad, end_rad, 0.f, grid);
  fftN.reverse(grid);

  Cropper crop(info, gridder->mapping().cartDims, fov);
  Cx4 channels = crop.crop4(grid);
  if (dim[0] == 1) {
    // setOnes() does not seem to work
    channels.setConstant(1); // PSF to fix single channel SENSE estimation
  }
  Cx3 rss = crop.newImage();
  rss.device(Threads::GlobalDevice()) = ConjugateSum(channels, channels).sqrt();
  if (lambda) {
    Log::Print(FMT_STRING("Regularization lambda {}"), lambda);
    rss.device(Threads::GlobalDevice()) = rss + rss.constant(lambda);
  }
  Log::Image(rss, "sense-rss.nii");
  Log::Image(channels, "sense-channels.nii");
  Log::Print(FMT_STRING("Normalizing channel images"));
  channels.device(Threads::GlobalDevice()) = channels / TileToMatch(rss, channels.dimensions());
  Log::Image(channels, "sense-maps.nii");
  Log::Print(FMT_STRING("Finished SENSE maps"));
  return channels;
}

Cx4 LoadSENSE(std::string const &calFile)
{
  HD5::Reader senseReader(calFile);
  return senseReader.readTensor<Cx4>(HD5::Keys::SENSE);
}

Cx4 InterpSENSE(std::string const &file, Eigen::Array3l const dims)
{
  HD5::Reader senseReader(file);
  Cx4 disk_sense = senseReader.readTensor<Cx4>(HD5::Keys::SENSE);
  Log::Print(FMT_STRING("Interpolating SENSE maps to dimensions {}"), dims.transpose());
  FFT::Planned<4, 3> fft1(disk_sense.dimensions());
  fft1.forward(disk_sense);
  Sz3 size1{disk_sense.dimension(1), disk_sense.dimension(2), disk_sense.dimension(3)};
  Sz3 size2{dims[0], dims[1], dims[2]};
  Cx4 sense(disk_sense.dimension(0), dims[0], dims[1], dims[2]);
  FFT::Planned<4, 3> fft2(sense);
  sense.setZero();
  if (size1[0] < size2[0]) {
    Crop4(sense, size1) = disk_sense;
  } else {
    sense = Crop4(disk_sense, size2);
  }
  fft2.reverse(sense);
  return sense;
}