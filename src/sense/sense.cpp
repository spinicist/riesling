#include "sense/sense.hpp"

#include "cropper.h"
#include "espirit.hpp"
#include "fft/fft.hpp"
#include "filter.hpp"
#include "io/hd5.hpp"
#include "op/grid.hpp"
#include "op/loop.hpp"
#include "op/tensorscale.hpp"
#include "sdc.hpp"
#include "tensorOps.hpp"
#include "threads.hpp"

namespace rl {
namespace SENSE {

Opts::Opts(args::Subparser &parser)
  : type(parser, "T", "SENSE type (auto/espirit/file.h5)", {"sense", 's'}, "auto")
  , volume(parser, "V", "SENSE calibration volume (first)", {"sense-vol"}, 0)
  , frame(parser, "F", "SENSE calibration frame (first)", {"sense-frame"}, 0)
  , res(parser, "R", "SENSE calibration res (12 mm)", {"sense-res"}, 12.f)
  , λ(parser, "L", "SENSE regularization", {"sense-lambda"}, 0.f)
  , fov(parser, "F", "SENSE FoV (default 256mm)", {"sense-fov"}, 256)
  , kRad(parser, "K", "ESPIRIT kernel size (4)", {"espirit-k"}, 3)
  , calRad(parser, "C", "ESPIRIT calibration region (8)", {"espirit-cal"}, 6)
  , gap(parser, "G", "ESPIRIT gap (0)", {"espirit-gap"}, 0)
  , threshold(parser, "T", "ESPIRIT retention threshold (0.015)", {"espirit-thresh"}, 0.015f)
{
}

auto LoresGrid(Opts &opts, CoreOpts &coreOpts, Trajectory const &inTraj, HD5::Reader &reader)
  -> Cx4
{
  Log::Print("SENSE Self-Calibration Starting");
  auto const nV = reader.dimensions<5>(HD5::Keys::Noncartesian)[4];
  if (opts.volume.Get() >= nV) {
    Log::Fail("Specified SENSE volume {} is greater than number of volumes in data {}", opts.volume.Get(), nV);
  }
  Cx4 const data = reader.readSlab<Cx4>(HD5::Keys::Noncartesian, opts.volume.Get());
  auto const nC = data.dimension(0);

  auto const [traj, lo, sz] = inTraj.downsample(opts.res.Get(), 0, false, true);
  auto sdcW = traj.nDims() == 2 ? SDC::Pipe<2>(traj) : SDC::Pipe<3>(traj);
  auto sdc3 = std::make_shared<TensorScale<Cx, 3>>(Sz3{nC, traj.nSamples(), traj.nTraces()}, sdcW.cast<Cx>());
  auto sdc = std::make_shared<LoopOp<TensorScale<Cx, 3>>>(sdc3, data.dimension(3));
  auto grid = make_3d_grid(traj, coreOpts.ktype.Get(), coreOpts.osamp.Get(), nC);

  Cx4 lores = data.slice(Sz4{0, lo, 0, 0}, Sz4{nC, sz, data.dimension(2), data.dimension(3)});
  auto const maxCoord = Maximum(NoNaNs(traj.points()).abs());
  NoncartesianTukey(maxCoord * 0.75, maxCoord, 0.f, traj.points(), lores);
  return grid->adjoint(sdc->adjoint(lores)).chip<1>(opts.frame.Get());
}

auto UniformNoise(float const λ, Sz3 const shape, Cx4 &channels) -> Cx4
{
  // FFT and then crop
  auto fft = FFT::Make<4, 3>(channels.dimensions());
  fft->reverse(channels);
  Cx4 cropped = Crop(channels, AddFront(shape, channels.dimension(0)));
  Cx3 rss(LastN<3>(cropped.dimensions()));
  rss.device(Threads::GlobalDevice()) = ConjugateSum(cropped, cropped).sqrt();
  if (λ > 0.f) {
    Log::Print("SENSE λ {}", λ);
    rss.device(Threads::GlobalDevice()) = rss + rss.constant(λ);
  }
  Log::Print<Log::Level::High>("Normalizing channel images");
  cropped.device(Threads::GlobalDevice()) = cropped / TileToMatch(rss, cropped.dimensions());
  return cropped;
}

Cx4 Choose(Opts &opts, CoreOpts &core, Trajectory const &traj, HD5::Reader &reader)
{
  Sz3 const shape = traj.matrix(opts.fov.Get());
  Log::Print("{}", opts.type.Get());
  if (opts.type.Get() == "auto") {
    Cx4 channels = LoresGrid(opts, core, traj, reader);
    return UniformNoise(opts.λ.Get(), shape, channels);
  } else if (opts.type.Get() == "espirit") {
    auto grid = LoresGrid(opts, core, traj, reader);
    return ESPIRIT(grid, shape, opts.kRad.Get(), opts.calRad.Get(), opts.gap.Get(), opts.threshold.Get());
  } else {
    HD5::Reader senseReader(opts.type.Get());
    Cx4 sense = senseReader.readTensor<Cx4>(HD5::Keys::SENSE);
    if (LastN<3>(sense.dimensions()) != shape) {
      Log::Fail("SENSE map spatial dimensions were {}, expected {}", LastN<3>(sense.dimensions()), shape);
    }
    return sense;
  }
}

} // namespace SENSE
} // namespace rl
