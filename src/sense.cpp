#include "sense.h"

#include "cropper.h"
#include "fft/fft.hpp"
#include "filter.hpp"
#include "io/hd5.hpp"
#include "tensorOps.hpp"
#include "threads.hpp"

namespace rl {
namespace SENSE {

Opts::Opts(args::Subparser &parser)
  : file(parser, "F", "Read SENSE maps from .h5", {"sense", 's'})
  , volume(parser, "V", "SENSE calibration volume (last)", {"sense-vol"}, -1)
  , frame(parser, "F", "SENSE calibration frame (first)", {"sense-frame"}, 0)
  , res(parser, "R", "SENSE calibration res (12 mm)", {"sense-res"}, 12.f)
  , λ(parser, "L", "SENSE regularization", {"sense-lambda"}, 0.f)
{
}

Cx4 SelfCalibration(
  Info const &info,
  GridBase<Cx, 3> *gridder,
  float const fov,
  float const res,
  float const λ,
  Index const frame,
  Cx3 const &data)
{
  Log::Print(FMT_STRING("SENSE Self-Calibration Starting"));
  Sz5 const dims = gridder->inputDimensions();
  Cropper crop(info, LastN<3>(dims), fov);
  Cx4 channels(crop.dims(dims[0]));
  if (dims[0] == 1) { // Only one channel, return all ones
    channels.setConstant(1.);
    return channels;
  }

  Cx4 grid(dims[0], dims[2], dims[3], dims[4]);
  if (frame >= info.frames) {
    Log::Fail("Specified SENSE frame {} is greater than number of frames in data {}", frame, info.frames);
  }
  grid = gridder->adjoint(data).chip<1>(frame);
  float const end_rad = info.voxel_size.minCoeff() / res;
  float const start_rad = 0.5 * end_rad;
  Log::Print<Log::Level::High>(FMT_STRING("SENSE resultion {} filter indices {}-{}"), res, start_rad, end_rad);
  KSTukey(start_rad, end_rad, 0.f, grid);
  auto const fft = FFT::Make<4, 3>(grid.dimensions());
  fft->reverse(grid);
  channels = crop.crop4(grid);

  Cx3 rss = crop.newImage();
  rss.device(Threads::GlobalDevice()) = ConjugateSum(channels, channels).sqrt();
  if (λ > 0.f) {
    Log::Print(FMT_STRING("Regularization lambda {}"), λ);
    rss.device(Threads::GlobalDevice()) = rss + rss.constant(λ);
  }
  Log::Print<Log::Level::High>(FMT_STRING("Normalizing channel images"));
  channels.device(Threads::GlobalDevice()) = channels / TileToMatch(rss, channels.dimensions());
  Log::Print(FMT_STRING("SENSE Self-Calibration finished"));
  return channels;
}

Cx4 Load(std::string const &calFile, Info const &i)
{
  HD5::Reader senseReader(calFile);
  auto const maps = senseReader.readTensor<Cx4>(HD5::Keys::SENSE);
  if (maps.dimension(0) != i.channels) {
    Log::Fail("SENSE maps had {} channels, should be {}", maps.dimension(0), i.channels);
  }
  return maps;
}

Cx4 Interp(std::string const &file, Sz3 const size2)
{
  HD5::Reader senseReader(file);
  Cx4 disk_sense = senseReader.readTensor<Cx4>(HD5::Keys::SENSE);
  Log::Print(FMT_STRING("Interpolating SENSE maps to dimensions {}"), size2);
  auto const fft1 = FFT::Make<4, 3>(disk_sense.dimensions());
  fft1->forward(disk_sense);
  Sz3 size1{disk_sense.dimension(1), disk_sense.dimension(2), disk_sense.dimension(3)};
  Cx4 sense(disk_sense.dimension(0), size2[0], size2[1], size2[2]);
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

Cx4 Choose(
  Opts &opts, Info const &info, GridBase<Cx, 3> *gridder, float const fov, SDCOp *sdc, HD5::RieslingReader &reader)
{
  if (gridder->inputDimensions()[0] == 1) { // Only one channel, return all ones
    Sz5 const dims = gridder->inputDimensions();
    Cropper crop(info, LastN<3>(dims), fov);
    Cx4 channels(crop.dims(dims[0]));
    channels.setConstant(1.);
    return channels;
  } else if (opts.file) {
    return Load(opts.file.Get(), info);
  } else {
    return SelfCalibration(
      info,
      gridder,
      fov,
      opts.res.Get(),
      opts.λ.Get(),
      opts.frame.Get(),
      sdc->adjoint(reader.noncartesian(ValOrLast(opts.volume.Get(), reader.trajectory().info().volumes))));
  }
}

} // namespace SENSE
} // namespace rl
