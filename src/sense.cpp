#include "sense.h"

#include "cropper.h"
#include "fft/fft.hpp"
#include "filter.h"
#include "io/io.h"
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
  Cropper crop(info, LastN<3>(dims), fov);
  Cx4 channels(crop.dims(dims[0]));
  if (dims[0] == 1) { // Only one channel, return all ones
    channels.setConstant(1.);
    return channels;
  }

  Cx4 grid(dims[0], dims[2], dims[3], dims[4]);
  grid = gridder->Adj(data).chip<1>(0); // Assume we want the first frame only
  float const end_rad = info.voxel_size.minCoeff() / res;
  float const start_rad = 0.5 * end_rad;
  Log::Print(FMT_STRING("SENSE res {} filter {}-{}"), res, start_rad, end_rad);
  KSTukey(start_rad, end_rad, 0.f, grid);
  auto const fft = FFT::Make<4, 3>(grid.dimensions());
  fft->reverse(grid);
  channels = crop.crop4(grid);

  Cx3 rss = crop.newImage();
  rss.device(Threads::GlobalDevice()) = ConjugateSum(channels, channels).sqrt();
  if (lambda) {
    Log::Print(FMT_STRING("Regularization lambda {}"), lambda);
    rss.device(Threads::GlobalDevice()) = rss + rss.constant(lambda);
  }
  Log::Image(rss, "sense-rss");
  Log::Image(channels, "sense-channels");
  Log::Print(FMT_STRING("Normalizing channel images"));
  channels.device(Threads::GlobalDevice()) = channels / TileToMatch(rss, channels.dimensions());
  Log::Image(channels, "sense-maps");
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
  auto const fft1 = FFT::Make<4, 3>(disk_sense.dimensions());
  fft1->forward(disk_sense);
  Sz3 size1{disk_sense.dimension(1), disk_sense.dimension(2), disk_sense.dimension(3)};
  Sz3 size2{dims[0], dims[1], dims[2]};
  Cx4 sense(disk_sense.dimension(0), dims[0], dims[1], dims[2]);
  auto const fft2 = FFT::Make<4, 3>(sense.dimensions());
  sense.setZero();
  if (size1[0] < size2[0]) {
    Crop4(sense, size1) = disk_sense;
  } else {
    sense = Crop4(disk_sense, size2);
  }
  fft2->reverse(sense);
  return sense;
}