#include "types.h"

#include "io_hd5.h"
#include "log.h"
#include "parse_args.h"

int main_split(args::Subparser &parser)
{
  args::Positional<std::string> fname(parser, "FILE", "HD5 file to recon");
  args::ValueFlag<std::string> oname(parser, "OUTPUT", "Override output name", {'o', "out"});
  args::ValueFlag<long> nspokes(parser, "NUM SPOKES", "Spokes per volume", {"n", "nspokes"}, 256);
  args::ValueFlag<long> vol(parser, "VOLUME", "Only take this volume", {"v", "vol"}, 0);
  args::ValueFlag<float> ds(parser, "DS", "Downsample by factor", {"ds"}, 1.0);
  // args::Flag nolo(parser, "NOLO", "Drop lowres spokes", {"nolo"});

  Log log = ParseCommand(parser, fname);

  HD5Reader reader(fname.Get(), log);
  auto const &in_info = reader.info();
  R3 traj = reader.readTrajectory();

  Cx3 rad_ks = in_info.noncartesianVolume();
  reader.readData(vol.Get(), rad_ks);

  Info out_info = in_info;

  Cx3 new_rad;
  R3 new_traj;

  // Remove lowres spokes for now
  new_rad = rad_ks.slice(
      Sz3{0, 0, in_info.spokes_lo}, Sz3{in_info.channels, in_info.read_points, in_info.spokes_hi});
  rad_ks = new_rad;

  new_traj =
      traj.slice(Sz3{0, 0, in_info.spokes_lo}, Sz3{3, in_info.read_points, in_info.spokes_hi});
  traj = new_traj;
  out_info.spokes_lo = 0;

  if (ds) {
    out_info.read_points = (long)std::round((float)in_info.read_points / ds.Get());
    float scalef = (float)in_info.read_points / (float)out_info.read_points;
    out_info.voxel_size = in_info.voxel_size * scalef;
    out_info.matrix = (in_info.matrix.cast<float>() / scalef).cast<long>();

    new_rad = rad_ks.slice(
        Sz3{0, 0, 0}, Sz3{out_info.channels, out_info.read_points, out_info.spokes_total()});
    rad_ks = new_rad;

    new_traj = traj.slice(Sz3{0, 0, 0}, Sz3{3, out_info.read_points, out_info.spokes_total()});
    traj = new_traj * new_traj.constant(ds.Get());
  }

  int num_int = static_cast<int>(out_info.spokes_total() * 1.f / nspokes.Get());
  log.info(FMT_STRING("Number of interleaves: {}"), num_int);
  int rem_spokes = out_info.spokes_total() - num_int * nspokes.Get();
  if (rem_spokes > 0) {
    log.info(FMT_STRING("Warning! Last interleave will have {} extra spokes."), rem_spokes);
  }
  int idx0;
  int idx1;
  int dspokes = nspokes.Get();
  for (int int_idx = 0; int_idx < num_int; int_idx++) {
    idx0 = nspokes.Get() * int_idx;

    if (int_idx == (num_int - 1)) {
      dspokes += rem_spokes;
    }
    HD5Writer writer(OutName(fname, oname, fmt::format("int{}", int_idx), "h5"), log);

    writer.writeData(
        0, rad_ks.slice(Sz3{0, 0, idx0}, Sz3{out_info.channels, out_info.read_points, dspokes}));
    writer.writeTrajectory(traj.slice(Sz3{0, 0, idx0}, Sz3{3, out_info.read_points, dspokes}));
    out_info.spokes_hi = dspokes;
    writer.writeInfo(out_info);
  }

  return EXIT_SUCCESS;
}