#include "types.h"

#include "io_hd5.h"
#include "log.h"
#include "parse_args.h"

int main_split(args::Subparser &parser)
{
  args::Positional<std::string> fname(parser, "FILE", "HD5 file to recon");
  args::ValueFlag<std::string> oname(parser, "OUTPUT", "Override output name", {'o', "out"});
  args::ValueFlag<long> nspokes(parser, "SPOKES", "Spokes per segment", {"n", "nspokes"});
  args::ValueFlag<long> vol(parser, "VOLUME", "Only take this volume", {"v", "vol"}, 0);
  args::ValueFlag<float> ds(parser, "DS", "Downsample by factor", {"ds"}, 1.0);

  Log log = ParseCommand(parser, fname);

  HD5::Reader reader(fname.Get(), log);
  auto const traj = reader.readTrajectory();
  auto const &info = reader.info();
  R3 all_points = traj.points();
  Cx3 all_ks = info.noncartesianVolume();
  reader.readNoncartesian(vol.Get(), all_ks);

  if (info.spokes_lo) {
    Info lo_info = info;
    lo_info.spokes_lo = 0;
    lo_info.spokes_hi = info.spokes_lo;
    lo_info.lo_scale = 0.f;
    R3 lo_points =
        all_points.slice(Sz3{0, 0, 0}, Sz3{3, info.read_points, info.spokes_lo}) / info.lo_scale;
    Cx4 lo_ks = all_ks.slice(Sz3{0, 0, 0}, Sz3{info.channels, info.read_points, info.spokes_lo})
                    .reshape(Sz4{info.channels, info.read_points, info.spokes_lo, 1});
    HD5::Writer writer(OutName(fname, oname, "lores", "h5"), log);
    writer.writeTrajectory(Trajectory(lo_info, lo_points, log));
    writer.writeNoncartesian(lo_ks);
  }

  Info hi_info = info;
  hi_info.spokes_lo = 0;
  hi_info.lo_scale = 0.f;
  if (ds) {
    hi_info.read_points = (long)std::round((float)info.read_points / ds.Get());
    float scalef = (float)info.read_points / (float)info.read_points;
    hi_info.voxel_size = info.voxel_size * scalef;
    hi_info.matrix = (info.matrix.cast<float>() / scalef).cast<long>();
  }
  int const ns = nspokes ? nspokes.Get() : info.spokes_hi;
  int const num_int = static_cast<int>(info.spokes_total() * 1.f / ns);
  log.info(FMT_STRING("Interleaves: {} Spokes per interleave: {}"), num_int, ns);
  int rem_spokes = info.spokes_hi - num_int * ns;
  if (rem_spokes > 0) {
    log.info(FMT_STRING("Warning! Last interleave will have {} extra spokes."), rem_spokes);
  }

  for (int int_idx = 0; int_idx < num_int; int_idx++) {
    int const idx0 = ns * int_idx + info.spokes_lo;
    int const n = ns + (int_idx == (num_int - 1) ? rem_spokes : 0);
    hi_info.spokes_hi = n;
    std::string const suffix = nspokes.Get() ? fmt::format("int{}", int_idx) : "hires";
    HD5::Writer writer(OutName(fname, oname, suffix, "h5"), log);
    writer.writeTrajectory(Trajectory(
        hi_info, all_points.slice(Sz3{0, 0, idx0}, Sz3{3, hi_info.read_points, n}), log));
    writer.writeNoncartesian(
        all_ks.slice(Sz3{0, 0, idx0}, Sz3{hi_info.channels, hi_info.read_points, n})
            .reshape(Sz4{hi_info.channels, hi_info.read_points, n, 1}));
  }

  return EXIT_SUCCESS;
}