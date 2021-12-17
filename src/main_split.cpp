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
  Info info = traj.info();
  info.volumes = 1; // Only output one volume
  R3 points = traj.points();
  I1 echoes = traj.echoes();
  Cx3 ks = reader.noncartesian(vol.Get());
  auto const volName = fmt::format("vol-{:02d}", vol.Get());

  if (ds) {
    info.read_points = (Index)std::round((float)info.read_points / ds.Get());
    float const scalef = (float)info.read_points / (float)info.read_points;
    info.voxel_size = info.voxel_size * scalef;
    info.matrix = (info.matrix.cast<float>() / scalef).cast<Index>();
    points = points * scalef;
    Cx3 ds_ks = ks.slice(Sz3{0, 0, 0}, Sz3{info.channels, info.read_points, info.spokes});
    ks = ds_ks;
  }

  if (lores) {
    log.info(FMT_STRING("Extracting {} low res spokes"), lores.Get());
    Info lo_info = info;
    lo_info.spokes = lores.Get();
    R3 lo_points = points.slice(Sz3{0, 0, 0}, Sz3{3, info.read_points, lores.Get()});
    Cx4 lo_ks = ks.slice(Sz3{0, 0, 0}, Sz3{info.channels, info.read_points, lores.Get()})
                  .reshape(Sz4{info.channels, info.read_points, lores.Get(), 1});
    I1 lo_echoes = echoes.slice(Sz1{0}, Sz1{lores.Get()});
    HD5::Writer writer(
      OutName(iname.Get(), oname.Get(), fmt::format("{}-lores", volName), "h5"), log);
    writer.writeTrajectory(Trajectory(lo_info, lo_points, lo_echoes, log));
    writer.writeNoncartesian(lo_ks);
    info.spokes -= lores.Get();
    log.info(FMT_STRING("Reduced spokes. {}, {}"), lores.Get(), info.spokes);
    log.info(FMT_STRING("points: {}"), points.dimensions());

    R3 hi_points = points.slice(Sz3{0, 0, lores.Get()}, Sz3{3, info.read_points, info.spokes});
    Cx3 hi_ks = ks.slice(Sz3{0, 0, lores.Get()}, Sz3{info.channels, info.read_points, info.spokes});
    I1 hi_echoes = echoes.slice(Sz1{lores.Get()}, Sz1{info.spokes});
    points = hi_points;
    ks = hi_ks;
    echoes = hi_echoes;
  }

  if (nspokes) {
    int const ns = nspokes.Get();
    int const spoke_step = step ? step.Get() : ns;
    int const num_full_int = static_cast<int>(info.spokes * 1.f / ns);
    int const num_int = static_cast<int>((num_full_int - 1) * ns * 1.f / spoke_step + 1);
    log.info(
      FMT_STRING("Interleaves: {} Spokes per interleave: {} Step: {}"), num_int, ns, spoke_step);
    log.info(FMT_STRING("info.spokes={}"), info.spokes);

    int rem_spokes = info.spokes - num_full_int * ns;
    if (rem_spokes > 0) {
      log.info(FMT_STRING("Warning! Last interleave will have {} extra spokes."), rem_spokes);
    }

    for (int int_idx = 0; int_idx < num_int; int_idx++) {
      int const idx0 = spoke_step * int_idx;
      int const n = ns + (int_idx == (num_int - 1) ? rem_spokes : 0);
      info.spokes = n;
      HD5::Writer writer(
        OutName(
          iname.Get(),
          oname.Get(),
          fmt::format(FMT_STRING("{}-int-{:02d}"), volName, int_idx),
          "h5"),
        log);
      writer.writeTrajectory(Trajectory(
        info,
        points.slice(Sz3{0, 0, idx0}, Sz3{3, info.read_points, n}),
        echoes.slice(Sz1{idx0}, Sz1{n}),
        log));
      writer.writeNoncartesian(ks.slice(Sz3{0, 0, idx0}, Sz3{info.channels, info.read_points, n})
                                 .reshape(Sz4{info.channels, info.read_points, n, 1}));
    }
  } else {
    HD5::Writer writer(OutName(iname.Get(), oname.Get(), volName, "h5"), log);
    writer.writeTrajectory(Trajectory(info, points, echoes, log));
    writer.writeNoncartesian(ks.reshape(Sz4{info.channels, info.read_points, info.spokes, 1}));
  }

  return EXIT_SUCCESS;
}