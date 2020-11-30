#include "io_hd5.h"
#include "io_nifti.h"
#include "log.h"
#include "parse_args.h"
#include "threads.h"
#include "types.h"
#include "zinfandel.h"
#include <filesystem>

int main_zinfandel(args::Subparser &parser)
{
  std::unordered_map<std::string, ZMode> z_map{{"1", ZMode::Z1_iter}, {"1s", ZMode::Z1_simul}};
  args::Positional<std::string> fname(parser, "INPUT FILE", "Input radial k-space to fill");
  args::ValueFlag<std::string> oname(
      parser, "OUTPUT NAME", "Name of output .h5 file", {"out", 'o'});
  args::ValueFlag<long> volume(parser, "VOLUME", "Only recon this volume", {"vol"}, -1);
  args::ValueFlag<long> gap(
      parser, "DEAD-TIME GAP", "Set gap value (default use header value)", {'g', "gap"}, -1);
  args::MapFlag<std::string, ZMode> z(
      parser,
      "Z TYPE",
      "Choose method (1 - iter, 1s - simul, 2 - non-lin iter, default iter)",
      {'z'},
      z_map,
      ZMode::Z1_iter);
  args::ValueFlag<long> z_src(
      parser, "Z SRC", "Number of ZINFANDEL sources (default 4)", {"zsrc"}, 4);
  args::ValueFlag<long> z_cal(
      parser, "Z CAL", "Size of calibration region (default all)", {"zcal"}, 0);
  args::ValueFlag<long> z_its(
      parser, "Z ITERS", "Maximum number of iterations (Z3 only)", {"zits"}, 10);
  args::ValueFlag<float> z_reg(parser, "Z REG", "Regularization lambda (default 0)", {"zreg"}, 0.f);
  args::ValueFlag<float> z_thresh(
      parser, "Z THRESH", "Threshold for SVD retention (Z3 only)", {"zthr"}, 0.1);
  Log log = ParseCommand(parser, fname);

  RadialReader reader(fname.Get(), log);
  auto info = reader.info();
  if (gap) {
    log.info(FMT_STRING("Setting dead-time gap to {} samples"), gap.Get());
    info.read_gap = gap.Get();
  }

  auto out_info = info;
  out_info.read_gap = 0;
  if (volume) {
    out_info.volumes = 1;
  }

  RadialWriter writer(OutName(fname, oname, "zinfandel", "h5"), log);
  writer.writeInfo(out_info);
  writer.writeTrajectory(reader.readTrajectory());
  writer.writeMeta(reader.readMeta());

  Cx3 rad_ks = info.radialVolume();
  for (auto const &iv : WhichVolumes(volume.Get(), info.volumes)) {
    reader.readData(iv, rad_ks);
    zinfandel(z.Get(), info.read_gap, z_src.Get(), z_cal.Get(), z_reg.Get(), rad_ks, log);
    writer.writeData(volume ? 0 : iv, rad_ks);
  }
  log.info("Finished");
  return EXIT_SUCCESS;
}
