#include "io.h"
#include "log.h"
#include "parse_args.h"
#include "slab_correct.h"
#include "threads.h"
#include "types.h"
#include "zinfandel.h"
#include <filesystem>

int main_zinfandel(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "INPUT FILE", "Input radial k-space to fill");
  args::ValueFlag<std::string> oname(
    parser, "OUTPUT NAME", "Name of output .h5 file", {"out", 'o'});
  args::ValueFlag<long> volume(parser, "VOLUME", "Only recon this volume", {"vol"}, -1);
  args::ValueFlag<long> gap(
    parser, "DEAD-TIME GAP", "Set gap value (default use header value)", {'g', "gap"}, -1);
  args::ValueFlag<long> src(
    parser, "SOURCES", "Number of ZINFANDEL sources (default 4)", {"src"}, 4);
  args::ValueFlag<long> spokes(
    parser, "CAL SPOKES", "Number of spokes to use for calibration (default 5)", {"spokes"}, 5);
  args::ValueFlag<long> read(
    parser, "CAL READ", "Read calibration size (default all)", {"read"}, 0);
  args::ValueFlag<float> l1(
    parser, "LAMBDA", "Tikhonov regularization (default 0)", {"lambda"}, 0.f);
  args::ValueFlag<float> pw(
    parser, "PULSE WIDTH", "Pulse-width for slab profile correction", {"pw"}, 0.f);
  args::ValueFlag<float> rbw(
    parser, "BANDWIDTH", "Read-out bandwidth for slab profile correction (kHz)", {"rbw"}, 0.f);
  Log log = ParseCommand(parser, iname);

  HD5::Reader reader(iname.Get(), log);
  auto info = reader.readInfo();
  long const gap_sz = gap ? gap.Get() : info.read_gap;
  auto const traj = reader.readTrajectory();
  auto out_info = info;
  out_info.read_gap = 0;
  if (volume) {
    out_info.volumes = 1;
  }

  HD5::Writer writer(OutName(iname.Get(), oname.Get(), "zinfandel", "h5"), log);
  writer.writeTrajectory(Trajectory(out_info, traj.points(), log));
  writer.writeMeta(reader.readMeta());

  Cx4 rad_ks = info.noncartesianSeries();
  for (long iv = 0; iv < info.volumes; iv++) {
    Cx3 vol = reader.noncartesian(iv);
    zinfandel(gap_sz, src.Get(), spokes.Get(), read.Get(), l1.Get(), traj.points(), vol, log);
    if (pw && rbw) {
      slab_correct(out_info, pw.Get(), rbw.Get(), vol, log);
    }
    rad_ks.chip(iv, 3) = vol;
  }
  writer.writeNoncartesian(rad_ks);
  log.info("Finished");
  return EXIT_SUCCESS;
}
