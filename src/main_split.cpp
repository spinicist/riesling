#include "types.h"

#include "io.h"
#include "log.h"
#include "parse_args.h"

int main_split(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "HD5 file to recon");
  args::ValueFlag<std::string> oname(parser, "OUTPUT", "Override output name", {'o', "out"});
  args::ValueFlag<Index> lores(parser, "N", "Extract first N spokes as lo-res", {'l', "lores"});
  args::ValueFlag<Index> nspokes(parser, "SPOKES", "Spokes per segment", {"n", "nspokes"});
  args::ValueFlag<Index> vol(parser, "VOLUME", "Only take this volume", {"v", "vol"}, 0);
  args::ValueFlag<float> ds(parser, "DS", "Downsample by factor", {"ds"}, 1.0);
  args::ValueFlag<Index> step(parser, "STEP", "Step size", {"s", "step"}, 0);

  Log log = ParseCommand(parser, iname);

  HD5::Reader reader(iname.Get(), log);
  auto const traj = reader.readTrajectory();
  auto info = traj.info();
  info.volumes = 1; // Only output one volume
  R3 points = traj.points();
  I1 echoes = traj.echoes();
  Cx3 ks = reader.noncartesian(vol.Get());

  if (ds) {
    info.read_points = (Index)std::round((float)info.read_points / ds.Get());
    float const scalef = (float)info.read_points / (float)info.read_points;
    info.voxel_size = info.voxel_size * scalef;
    info.matrix = (info.matrix.cast<float>() / scalef).cast<Index>();
    points = points * scalef;
    ks = ks.slice(Sz3{0, 0, 0}, Sz3{info.channels, info.read_points, info.spokes_total()});
  }

  if (lores) {
    Info lo_info = info;
    lo_info.spokes_lo = 0;
    lo_info.spokes_hi = lores.Get();
    lo_info.lo_scale = 0.f;
    R3 lo_points =
      points.slice(Sz3{0, 0, 0}, Sz3{3, info.read_points, lores.Get()}) / info.lo_scale;
    Cx4 lo_ks = ks.slice(Sz3{0, 0, 0}, Sz3{info.channels, info.read_points, lores.Get()})
                  .reshape(Sz4{info.channels, info.read_points, lores.Get(), 1});
    I1 lo_echoes = echoes.slice(Sz1{0}, Sz1{lores.Get()});
    HD5::Writer writer(OutName(iname.Get(), oname.Get(), "lores", "h5"), log);
    writer.writeTrajectory(Trajectory(lo_info, lo_points, lo_echoes, log));
    writer.writeNoncartesian(lo_ks);
    info.spokes_lo = 0;
    info.spokes_hi = info.spokes_total() - lores.Get();
    info.lo_scale = 0.f;
    points = points.slice(Sz3{0, lores.Get(), 0}, Sz3{3, info.read_points, info.spokes_hi});
    ks = ks.slice(Sz3{0, lores.Get(), 0}, Sz3{info.channels, info.read_points, info.spokes_hi});
    echoes = echoes.slice(Sz1{0}, Sz1{info.spokes_total() - lores.Get()});
  }

  if (nspokes) {
    int const ns = nspokes.Get();
    int const spoke_step = step ? step.Get() : ns;
    int const num_full_int = static_cast<int>(info.spokes_total() * 1.f / ns);
    int const num_int = static_cast<int>((num_full_int - 1) * ns * 1.f / spoke_step + 1);
    log.info(
      FMT_STRING("Interleaves: {} Spokes per interleave: {} Step: {}"), num_int, ns, spoke_step);
    int rem_spokes = info.spokes_hi - num_full_int * ns;
    if (rem_spokes > 0) {
      log.info(FMT_STRING("Warning! Last interleave will have {} extra spokes."), rem_spokes);
    }

    for (int int_idx = 0; int_idx < num_int; int_idx++) {
      int const idx0 = spoke_step * int_idx + info.spokes_lo;
      int const n = ns + (int_idx == (num_int - 1) ? rem_spokes : 0);
      info.spokes_hi = n;
      HD5::Writer writer(
        OutName(iname.Get(), oname.Get(), fmt::format("int{}", int_idx), "h5"), log);
      writer.writeTrajectory(Trajectory(
        info,
        points.slice(Sz3{0, 0, idx0}, Sz3{3, info.read_points, n}),
        echoes.slice(Sz1{idx0}, Sz1{n}),
        log));
      writer.writeNoncartesian(ks.slice(Sz3{0, 0, idx0}, Sz3{info.channels, info.read_points, n})
                                 .reshape(Sz4{info.channels, info.read_points, n, 1}));
    }
  } else {
    HD5::Writer writer(OutName(iname.Get(), oname.Get(), "hires", "h5"), log);
    writer.writeTrajectory(Trajectory(info, points, echoes, log));
    writer.writeNoncartesian(
      ks.reshape(Sz4{info.channels, info.read_points, info.spokes_total(), 1}));
  }

  return EXIT_SUCCESS;
}