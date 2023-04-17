#include "sense.hpp"

#include "cropper.h"
#include "io/hd5.hpp"
#include "op/nufft.hpp"
#include "sdc.hpp"
#include "tensorOps.hpp"
#include "threads.hpp"
#include "filter.hpp"

namespace rl {
namespace SENSE {

Opts::Opts(args::Subparser &parser)
  : file(parser, "F", "Read SENSE maps from .h5", {"sense", 's'})
  , volume(parser, "V", "SENSE calibration volume (first)", {"sense-vol"}, 0)
  , frame(parser, "F", "SENSE calibration frame (first)", {"sense-frame"}, 0)
  , res(parser, "R", "SENSE calibration res (12 mm)", {"sense-res"}, 12.f)
  , λ(parser, "L", "SENSE regularization", {"sense-lambda"}, 0.f)
  , fov(parser, "F", "SENSE FoV (default 256mm)", {"sense-fov"}, 256)
{
}

Cx4 SelfCalibration(Opts &opts, CoreOpts &coreOpts, Trajectory const &inTraj, HD5::Reader &reader)
{
  Log::Print(FMT_STRING("SENSE Self-Calibration Starting"));
  auto const nC = reader.dimensions<5>(HD5::Keys::Noncartesian)[0];
  auto const [traj, lo, sz] = inTraj.downsample(opts.res.Get(), 0, false);
  auto sdcW = traj.nDims() == 2 ? SDC::Pipe<2>(traj) : SDC::Pipe<3>(traj);
  auto sdc = std::make_shared<Scale<Cx, 3>>(Sz3{nC, traj.nSamples(), traj.nTraces()}, sdcW.cast<Cx>());
  auto nufft = make_nufft(traj, coreOpts.ktype.Get(), coreOpts.osamp.Get(), nC, traj.matrix(opts.fov.Get()), std::nullopt, sdc);
  Sz5 const dims = nufft->ishape;
  Cropper crop(traj.info().matrix, LastN<3>(dims), traj.info().voxel_size, opts.fov.Get());
  Cx4 channels(crop.dims(dims[0]));
  if (dims[0] == 1) { // Only one channel, return all ones
    channels.setConstant(1.);
    return channels;
  }

  auto const nV = reader.dimensions<5>(HD5::Keys::Noncartesian)[4];
  if (opts.volume.Get() >= nV) {
    Log::Fail("Specified SENSE volume {} is greater than number of volumes in data {}", opts.volume.Get(), nV);
  }

  Cx4 const data = reader.readSlab<Cx4>(HD5::Keys::Noncartesian, opts.volume.Get());
  Cx4 lores = data.slice(Sz4{0, lo, 0, 0}, Sz4{nC, sz, data.dimension(2), data.dimension(3)});
  auto const maxCoord = Maximum(NoNaNs(traj.points()).abs());
NoncartesianTukey(maxCoord * 0.75, maxCoord, 0.f, traj.points(), lores);
  Cx5 const allChan = nufft->adjoint(lores);
  channels = crop.crop5(allChan).chip<1>(opts.frame.Get());
  Cx3 rss = crop.newImage();
  rss.device(Threads::GlobalDevice()) = ConjugateSum(channels, channels).sqrt();
  if (opts.λ.Get() > 0.f) {
    Log::Print(FMT_STRING("Regularization lambda {}"), opts.λ.Get());
    rss.device(Threads::GlobalDevice()) = rss + rss.constant(opts.λ.Get());
  }
  Log::Print<Log::Level::High>(FMT_STRING("Normalizing channel images"));
  channels.device(Threads::GlobalDevice()) = channels / TileToMatch(rss, channels.dimensions());
  Log::Print(FMT_STRING("SENSE Self-Calibration finished"));
  return channels;
}

Cx4 Choose(Opts &opts, CoreOpts &core, Trajectory const &traj, HD5::Reader &reader)
{
  if (opts.file) {
    HD5::Reader senseReader(opts.file.Get());
    return senseReader.readTensor<Cx4>(HD5::Keys::SENSE);
  } else {
    return SelfCalibration(opts, core, traj, reader);
  }
}

} // namespace SENSE
} // namespace rl
