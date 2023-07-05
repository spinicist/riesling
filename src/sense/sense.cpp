#include "sense/sense.hpp"

#include "algo/lsmr.hpp"
#include "cropper.h"
#include "espirit.hpp"
#include "fft/fft.hpp"
#include "filter.hpp"
#include "io/hd5.hpp"
#include "op/nufft.hpp"
#include "precond.hpp"
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

auto LoresChannels(Opts &opts, CoreOpts &coreOpts, Trajectory const &inTraj, Cx5 const &noncart) -> Cx4
{
  Log::Print("SENSE Self-Calibration Starting");
  auto const nC = noncart.dimension(0);
  auto const nT = noncart.dimension(2);
  auto const nS = noncart.dimension(3);
  auto const nV = noncart.dimension(4);
  if (opts.volume.Get() >= nV) {
    Log::Fail("Specified SENSE volume {} is greater than number of volumes in data {}", opts.volume.Get(), nV);
  }

  auto const [traj, lo, sz] = inTraj.downsample(opts.res.Get(), 0, false, false);
  auto const maxCoord = Maximum(NoNaNs(traj.points()).abs());
  auto const nufft = make_nufft(traj, coreOpts.ktype.Get(), coreOpts.osamp.Get(), nC, traj.matrix(-1.f));
  auto const M = make_kspace_pre("kspace", nC, traj, IdBasis());
  LSMR const lsmr{nufft, M, 4};

  Cx4 lores = noncart.chip<4>(opts.volume.Get()).slice(Sz4{0, lo, 0, 0}, Sz4{nC, sz, nT, nS});
  NoncartesianTukey(maxCoord * 0.75, maxCoord, 0.f, traj.points(), lores);
  auto const channels = Tensorfy(lsmr.run(lores.data()), nufft->ishape);
  return channels.chip<1>(0);
}

auto UniformNoise(float const λ, Sz3 const shape, Cx4 &channels) -> Cx4
{
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

Cx4 Choose(Opts &opts, CoreOpts &core, Trajectory const &traj, Cx5 const &noncart)
{
  Sz3 const shape = traj.matrix(opts.fov.Get());
  if (opts.type.Get() == "auto") {
    Cx4 channels = LoresChannels(opts, core, traj, noncart);
    return UniformNoise(opts.λ.Get(), shape, channels);
  } else if (opts.type.Get() == "espirit") {
    auto channels = LoresChannels(opts, core, traj, noncart);
    auto fft = FFT::Make<4, 3>(channels.dimensions());
    fft->reverse(channels);
    return ESPIRIT(channels, shape, opts.kRad.Get(), opts.calRad.Get(), opts.gap.Get(), opts.threshold.Get());
  } else {
    HD5::Reader senseReader(opts.type.Get());
    Cx4         sense = senseReader.readTensor<Cx4>(HD5::Keys::SENSE);
    if (LastN<3>(sense.dimensions()) != shape) {
      Log::Fail("SENSE map spatial dimensions were {}, expected {}", LastN<3>(sense.dimensions()), shape);
    }
    return sense;
  }
}

} // namespace SENSE
} // namespace rl
