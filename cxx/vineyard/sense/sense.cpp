#include "sense/sense.hpp"

#include "algo/lsmr.hpp"
#include "cropper.hpp"
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
  , res(parser, "R", "SENSE calibration res (12 mm)", {"sense-res"}, Eigen::Array3f::Constant(12.f))
  , fov(parser, "SENSE-FOV", "SENSE FoV (default 256,256,256)", {"sense-fov"}, Eigen::Array3f::Zero())
  , λ(parser, "L", "SENSE regularization", {"sense-lambda"}, 0.f)
  , kRad(parser, "K", "ESPIRIT kernel size (4)", {"espirit-k"}, 3)
  , calRad(parser, "C", "ESPIRIT calibration region (8)", {"espirit-cal"}, 6)
  , gap(parser, "G", "ESPIRIT gap (0)", {"espirit-gap"}, 0)
  , threshold(parser, "T", "ESPIRIT retention threshold (0.015)", {"espirit-thresh"}, 0.015f)
{
}

auto LoresChannels(Opts &opts, GridOpts &gridOpts, Trajectory const &inTraj, Cx5 const &noncart, Basis<Cx> const &basis) -> Cx5
{
  auto const nC = noncart.dimension(0);
  auto const nT = noncart.dimension(2);
  auto const nS = noncart.dimension(3);
  auto const nV = noncart.dimension(4);
  if (opts.volume.Get() >= nV) {
    Log::Fail("Specified SENSE volume {} is greater than number of volumes in data {}", opts.volume.Get(), nV);
  }

  auto const [traj, lo, sz] = inTraj.downsample(opts.res.Get(), 0, false, false);
  auto const nufft = make_nufft(traj, gridOpts, nC, traj.matrixForFOV(opts.fov.Get()), basis);
  auto const M = make_kspace_pre(traj, nC, basis);
  LSMR const lsmr{nufft, M, 4};

  Cx4        lores = noncart.chip<4>(opts.volume.Get()).slice(Sz4{0, lo, 0, 0}, Sz4{nC, sz, nT, nS});
  auto const maxCoord = Maximum(NoNaNs(traj.points()).abs());
  NoncartesianTukey(maxCoord * 0.75, maxCoord, 0.f, traj.points(), lores);
  Cx5 const channels(Tensorfy(lsmr.run(lores.data()), nufft->ishape));

  Sz3 const shape = traj.matrixForFOV(opts.fov.Get());
  for (Index ii = 0; ii < 3; ii++) {
    if (shape[ii] > channels.dimension(ii + 2)) {
      Log::Fail("Requested SENSE FOV {} could not be satisfied with FOV {} and oversampling {}", opts.fov.Get().transpose(),
                traj.FOV().transpose(), gridOpts.osamp.Get());
    }
  }

  Cx5 const cropped = Crop(channels, AddFront(shape, channels.dimension(0), channels.dimension(1)));

  return cropped;
}

void RegularizedNormalization(float const λ, Cx4 const &ref, Cx5 &channels)
{
  Sz5 const shape = channels.dimensions();
  Log::Debug("Normalizing SENSE. Dimensions {} λ {}", shape, λ);
  Cx5 normalized(shape);
  channels.device(Threads::GlobalDevice()) =
    channels / (ref + ref.constant(λ)).reshape(AddFront(LastN<4>(shape), 1)).broadcast(Sz5{shape[0], 1, 1, 1, 1});
}

void RegularizedNormalization(float const λ, Cx5 &channels)
{
  Cx4 rss(LastN<4>(channels.dimensions()));
  rss.device(Threads::GlobalDevice()) = ConjugateSum(channels, channels).sqrt();
  RegularizedNormalization(λ, rss, channels);
}

auto Choose(Opts &opts, GridOpts &nufft, Trajectory const &traj, Cx5 const &noncart) -> Cx5
{
  if (opts.type.Get() == "auto") {
    Log::Print("SENSE Self-Calibration");
    auto c = LoresChannels(opts, nufft, traj, noncart);
    RegularizedNormalization(opts.λ.Get(), c);
    return c;
  } else if (opts.type.Get() == "espirit") {
    Log::Fail("Not supported right now");
    // auto channels = LoresChannels(opts, core, traj, noncart);
    // auto fft = FFT::Make<5, 3>(channels.dimensions());
    // fft->reverse(channels);
    // return ESPIRIT(channels, shape, opts.kRad.Get(), opts.calRad.Get(), opts.gap.Get(), opts.threshold.Get());
  } else {
    HD5::Reader senseReader(opts.type.Get());
    return senseReader.readTensor<Cx5>(HD5::Keys::Data);
  }
}

} // namespace SENSE
} // namespace rl
