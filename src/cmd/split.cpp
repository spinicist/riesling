#include "types.h"

#include "io/hd5.hpp"
#include "log.h"
#include "parse_args.h"
#include "tensorOps.h"

#include <numeric>

int main_split(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "HD5 file to recon");
  args::ValueFlag<std::string> oname(parser, "OUTPUT", "Override output name", {'o', "out"});
  args::ValueFlag<Index> lores(parser, "N", "Extract N spokes as lo-res (negative at end)", {'l', "lores"}, 0);
  args::ValueFlag<Index> spoke_stride(parser, "S", "Hi-res stride", {"stride"}, 1);
  args::ValueFlag<Index> spoke_size(parser, "SZ", "Size of hi-res spokes to keep", {"size"});
  args::ValueFlag<Index> spi(parser, "SPOKES", "Spokes per interleave", {"spi"});
  args::ValueFlag<Index> nF(parser, "F", "Break into N frames", {"frames"}, 1);
  args::ValueFlag<Index> spf(parser, "S", "Spokes per frame", {"spf"}, 1);
  args::ValueFlag<Index> step(parser, "STEP", "Step size", {"s", "step"}, 0);
  args::ValueFlag<Index> zero(parser, "Z", "Zero the first N samples", {"zero"}, 0);
  args::ValueFlag<Index> trim(parser, "T", "Trim the first N samples", {"trim"}, 0);
  args::ValueFlag<Index> vol(parser, "V", "Take this volume", {"vol"});

  ParseCommand(parser, iname);

  HD5::RieslingReader reader(iname.Get());
  auto traj = reader.trajectory();
  Cx4 ks = reader.readTensor<Cx4>(HD5::Keys::Noncartesian);

  if (vol) {
    auto info = traj.info();
    if (vol.Get() >= info.volumes) {
      Log::Fail("Specified volume {} past end of file", vol.Get());
    }
    Log::Print("Selecting volume {}", vol.Get());
    info.volumes = 1;
    traj = Trajectory(info, traj.points(), traj.frames());
    ks = Cx4(ks.slice(Sz4{0, 0, 0, vol.Get()}, Sz4{info.channels, info.read_points, info.spokes, 1}));
  }

  if (trim) {
    Log::Print(FMT_STRING("Trimming {} points"), trim.Get());
    auto info = traj.info();
    info.read_points = info.read_points - trim.Get();
    R3 points = traj.points();
    points = R3(points.slice(Sz3{0, trim.Get(), 0}, Sz3{3, info.read_points, info.spokes}));
    traj = Trajectory(info, points, traj.frames());
    ks = Cx4(ks.slice(Sz4{0, trim.Get(), 0, 0}, Sz4{info.channels, info.read_points, info.spokes, info.volumes}));
  }

  if (zero) {
    Log::Print(FMT_STRING("Zeroing {} popints"), zero.Get());
    ks.slice(Sz4{0, 0, 0, 0}, Sz4{ks.dimension(0), zero.Get(), ks.dimension(2), ks.dimension(3)}).setZero();
  }

  if (lores) {
    if (lores.Get() > traj.info().spokes) {
      Log::Fail(FMT_STRING("Invalid number of low-res spokes {}"), lores.Get());
    }

    auto info = traj.info();
    Index const spokesLo = std::abs(lores.Get());
    Index const spokesHi = info.spokes - spokesLo;
    bool atEnd = (lores.Get() < 0);
    I1 const lo_frames = traj.frames().slice(Sz1{atEnd ? spokesHi : 0}, Sz1{spokesLo});
    Log::Print(FMT_STRING("Extracting spokes {}-{} as low-res"), atEnd ? spokesHi : 0, spokesLo);

    Info lo_info = traj.info();
    lo_info.spokes = spokesLo;
    Trajectory lo_traj(
      lo_info, traj.points().slice(Sz3{0, 0, atEnd ? spokesHi : 0}, Sz3{3, lo_info.read_points, spokesLo}), lo_frames);
    Cx4 lo_ks = ks.slice(
      Sz4{0, 0, atEnd ? spokesHi : 0, 0}, Sz4{lo_info.channels, lo_info.read_points, spokesLo, lo_info.volumes});

    info.spokes = spokesHi;
    I1 const hi_frames(traj.frames().slice(Sz1{atEnd ? 0 : spokesLo}, Sz1{spokesHi}));
    traj = Trajectory(
      info, R3(traj.points().slice(Sz3{0, 0, atEnd ? 0 : spokesLo}, Sz3{3, info.read_points, spokesHi})), hi_frames);
    ks =
      Cx4(ks.slice(Sz4{0, 0, atEnd ? 0 : spokesLo, 0}, Sz4{info.channels, info.read_points, spokesHi, info.volumes}));

    HD5::Writer writer(OutName(iname.Get(), oname.Get(), "lores"));
    writer.writeTrajectory(lo_traj);
    writer.writeTensor(lo_ks, HD5::Keys::Noncartesian);
  }

  if (nF && spf) {
    Index const sps = spf.Get() * nF.Get();
    if (traj.info().spokes % spf != 0) {
      Log::Fail(FMT_STRING("SPE {} does not divide spokes {} cleanly"), sps, traj.info().spokes);
    }
    Index const segs = std::ceil(static_cast<float>(traj.info().spokes) / sps);
    Log::Print(
      FMT_STRING("Adding info for {} frames with {} spokes per frame, {} per segment, {} segments"),
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
                  .slice(Sz1{0}, Sz1{traj.info().spokes});
    Info info = traj.info();
    info.frames = nF.Get();
    traj = Trajectory(info, traj.points(), frames);
  }

  if (spoke_stride) {
    auto info = traj.info();
    ks = Cx4(ks.stride(Sz4{1, 1, spoke_stride.Get(), 1}));
    info.spokes = ks.dimension(2);
    traj = Trajectory(
      info, traj.points().stride(Sz3{1, 1, spoke_stride.Get()}), traj.frames().stride(Sz1{spoke_stride.Get()}));
  }

  if (spoke_size) {
    auto info = traj.info();
    info.spokes = spoke_size.Get();
    ks = Cx4(ks.slice(Sz4{0, 0, 0, 0}, Sz4{info.channels, info.read_points, info.spokes, info.volumes}));
    traj = Trajectory(
      info,
      traj.points().slice(Sz3{0, 0, 0}, Sz3{3, info.read_points, info.spokes}),
      traj.frames().slice(Sz1{0}, Sz1{info.spokes}));
  }

  if (spi) {
    auto info = traj.info();
    int const ns = spi.Get();
    int const spoke_step = step ? step.Get() : ns;
    int const num_full_int = static_cast<int>(info.spokes * 1.f / ns);
    int const num_int = static_cast<int>((num_full_int - 1) * ns * 1.f / spoke_step + 1);
    Log::Print(FMT_STRING("Interleaves: {} Spokes per interleave: {} Step: {}"), num_int, ns, spoke_step);
    int rem_spokes = info.spokes - num_full_int * ns;
    if (rem_spokes > 0) {
      Log::Print(FMT_STRING("Warning! Last interleave will have {} extra spokes."), rem_spokes);
    }

    for (int int_idx = 0; int_idx < num_int; int_idx++) {
      int const idx0 = spoke_step * int_idx;
      int const n = ns + (int_idx == (num_int - 1) ? rem_spokes : 0);
      info.spokes = n;
      HD5::Writer writer(OutName(iname.Get(), oname.Get(), fmt::format(FMT_STRING("hires-{:02d}"), int_idx)));
      writer.writeTrajectory(Trajectory(
        info,
        traj.points().slice(Sz3{0, 0, idx0}, Sz3{3, info.read_points, n}),
        traj.frames().slice(Sz1{idx0}, Sz1{n})));
      writer.writeTensor(
        Cx4(ks.slice(Sz4{0, 0, idx0, 0}, Sz4{info.channels, info.read_points, n, info.volumes})),
        HD5::Keys::Noncartesian);
    }
  } else {
    HD5::Writer writer(OutName(iname.Get(), oname.Get(), "hires"));
    writer.writeTrajectory(traj);
    writer.writeTensor(ks, HD5::Keys::Noncartesian);
  }

  return EXIT_SUCCESS;
}