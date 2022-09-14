#include "types.h"

#include "io/hd5.hpp"
#include "log.hpp"
#include "parse_args.hpp"
#include "tensorOps.hpp"

using namespace rl;

int main_split(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "HD5 file to recon");
  args::ValueFlag<std::string> oname(parser, "OUTPUT", "Override output name", {'o', "out"});
  args::ValueFlag<Index> lores(parser, "N", "Extract N traces as lo-res (negative at end)", {'l', "lores"}, 0);
  args::ValueFlag<Index> spoke_stride(parser, "S", "Hi-res stride", {"stride"}, 1);
  args::ValueFlag<Index> spoke_size(parser, "SZ", "Size of hi-res traces to keep", {"size"});
  args::ValueFlag<Index> spi(parser, "traces", "traces per interleave", {"spi"});
  args::ValueFlag<Index> nF(parser, "F", "Break into N frames", {"frames"}, 1);
  args::ValueFlag<Index> spf(parser, "S", "traces per frame", {"spf"}, 1);
  args::ValueFlag<Index> step(parser, "STEP", "Step size", {"s", "step"}, 0);
  args::ValueFlag<Index> zero(parser, "Z", "Zero the first N samples", {"zero"}, 0);
  args::ValueFlag<Index> trim(parser, "T", "Trim the first N samples", {"trim"}, 0);
  args::ValueFlag<Index> vol(parser, "V", "Take this volume", {"vol"});

  ParseCommand(parser, iname);

  HD5::RieslingReader reader(iname.Get());
  auto traj = reader.trajectory();
  Cx4 ks = reader.readTensor<Cx4>(HD5::Keys::Noncartesian);
  Index const channels = ks.dimension(0);
  Index volumes = ks.dimension(3);
  if (vol) {
    auto info = traj.info();
    if (vol.Get() >= volumes) {
      Log::Fail("Specified volume {} past end of file", vol.Get());
    }
    Log::Print("Selecting volume {}", vol.Get());
    volumes = 1;
    traj = Trajectory(info, traj.points(), traj.nFrames());
    ks = Cx4(ks.slice(Sz4{0, 0, 0, vol.Get()}, AddBack(FirstN<3>(ks.dimensions()), 1)));
  }

  if (trim) {
    Log::Print(FMT_STRING("Trimming {} points"), trim.Get());
    auto info = traj.info();
    Re3 points = traj.points();
    points = Re3(points.slice(Sz3{0, trim.Get(), 0}, Sz3{3, points.dimension(1) - trim.Get(), points.dimension(2)}));
    traj = Trajectory(info, points, traj.nFrames());
    ks = Cx4(ks.slice(Sz4{0, trim.Get(), 0, 0}, Sz4{channels, points.dimension(1), ks.dimension(2), volumes}));
  }

  if (zero) {
    Log::Print(FMT_STRING("Zeroing {} popints"), zero.Get());
    ks.slice(Sz4{0, 0, 0, 0}, Sz4{ks.dimension(0), zero.Get(), ks.dimension(2), ks.dimension(3)}).setZero();
  }

  if (lores) {
    if (lores.Get() > traj.nTraces()) {
      Log::Fail(FMT_STRING("Invalid number of low-res traces {}"), lores.Get());
    }

    auto info = traj.info();
    Index const tracesLo = std::abs(lores.Get());
    Index const tracesHi = traj.nTraces() - tracesLo;
    bool atEnd = (lores.Get() < 0);
    I1 const lo_frames = traj.frames().slice(Sz1{atEnd ? tracesHi : 0}, Sz1{tracesLo});
    Log::Print(FMT_STRING("Extracting traces {}-{} as low-res"), atEnd ? tracesHi : 0, tracesLo);

    Info lo_info = traj.info();
    Trajectory lo_traj(
      lo_info, traj.points().slice(Sz3{0, 0, atEnd ? tracesHi : 0}, Sz3{3, traj.nSamples(), tracesLo}), lo_frames);
    Cx4 lo_ks = ks.slice(
      Sz4{0, 0, atEnd ? tracesHi : 0, 0}, Sz4{channels, traj.nSamples(), tracesLo, volumes});

    I1 const hi_frames(traj.frames().slice(Sz1{atEnd ? 0 : tracesLo}, Sz1{tracesHi}));
    traj = Trajectory(
      info, Re3(traj.points().slice(Sz3{0, 0, atEnd ? 0 : tracesLo}, Sz3{3, traj.nSamples(), tracesHi})), hi_frames);
    ks =
      Cx4(ks.slice(Sz4{0, 0, atEnd ? 0 : tracesLo, 0}, Sz4{channels, traj.nSamples(), tracesHi, volumes}));

    HD5::Writer writer(OutName(iname.Get(), oname.Get(), "lores"));
    writer.writeTrajectory(lo_traj);
    writer.writeTensor(lo_ks, HD5::Keys::Noncartesian);
  }

  if (nF && spf) {
    Index const sps = spf.Get() * nF.Get();
    if (traj.nTraces() % spf != 0) {
      Log::Fail(FMT_STRING("SPE {} does not divide traces {} cleanly"), sps, traj.nTraces());
    }
    Index const segs = std::ceil(static_cast<float>(traj.nTraces()) / sps);
    Log::Print(
      FMT_STRING("Adding info for {} frames with {} traces per frame, {} per segment, {} segments"),
      nF.Get(),
      spf.Get(),
      sps,
      segs);
    I1 e(nF.Get());
    std::iota(e.data(), e.data() + nF.Get(), 0);
    I1 frames = e.reshape(Sz2{1, nF.Get()})
                  .broadcast(Sz2{spf.Get(), 1})
                  .reshape(Sz1{sps})
                  .broadcast(Sz1{segs})
                  .slice(Sz1{0}, Sz1{traj.nTraces()});
    traj = Trajectory(traj.info(), traj.points(), frames);
  }

  if (spoke_stride) {
    ks = Cx4(ks.stride(Sz4{1, 1, spoke_stride.Get(), 1}));
    traj = Trajectory(
      traj.info(), traj.points().stride(Sz3{1, 1, spoke_stride.Get()}), traj.frames().stride(Sz1{spoke_stride.Get()}));
  }

  if (spoke_size) {
    auto info = traj.info();
    ks = Cx4(ks.slice(Sz4{0, 0, 0, 0}, Sz4{channels, traj.nSamples(), traj.nTraces(), volumes}));
    traj = Trajectory(
      info,
      traj.points().slice(Sz3{0, 0, 0}, Sz3{3, traj.nSamples(), traj.nTraces()}),
      traj.frames().slice(Sz1{0}, Sz1{traj.nTraces()}));
  }

  if (spi) {
    auto info = traj.info();
    int const ns = spi.Get();
    int const spoke_step = step ? step.Get() : ns;
    int const num_full_int = static_cast<int>(traj.nTraces() * 1.f / ns);
    int const num_int = static_cast<int>((num_full_int - 1) * ns * 1.f / spoke_step + 1);
    Log::Print(FMT_STRING("Interleaves: {} traces per interleave: {} Step: {}"), num_int, ns, spoke_step);
    int rem_traces = traj.nTraces() - num_full_int * ns;
    if (rem_traces > 0) {
      Log::Print(FMT_STRING("Warning! Last interleave will have {} extra traces."), rem_traces);
    }

    for (int int_idx = 0; int_idx < num_int; int_idx++) {
      int const idx0 = spoke_step * int_idx;
      int const n = ns + (int_idx == (num_int - 1) ? rem_traces : 0);
      HD5::Writer writer(OutName(iname.Get(), oname.Get(), fmt::format(FMT_STRING("hires-{:02d}"), int_idx)));
      writer.writeTrajectory(Trajectory(
        info,
        traj.points().slice(Sz3{0, 0, idx0}, Sz3{3, traj.nSamples(), n}),
        traj.frames().slice(Sz1{idx0}, Sz1{n})));
      writer.writeTensor(
        Cx4(ks.slice(Sz4{0, 0, idx0, 0}, Sz4{channels, traj.nSamples(), n, volumes})),
        HD5::Keys::Noncartesian);
    }
  } else {
    HD5::Writer writer(OutName(iname.Get(), oname.Get(), "hires"));
    writer.writeTrajectory(traj);
    writer.writeTensor(ks, HD5::Keys::Noncartesian);
  }

  return EXIT_SUCCESS;
}