#include "zinfandel.h"
#include "io/io.h"
#include "log.h"
#include "parse_args.h"
#include "threads.h"
#include "types.h"
#include <filesystem>

int main_zinfandel(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "INPUT FILE", "Input radial k-space to fill");
  args::ValueFlag<std::string> oname(
    parser, "OUTPUT NAME", "Name of output .h5 file", {"out", 'o'});
  args::ValueFlag<Index> volume(parser, "VOLUME", "Only recon this volume", {"vol"}, -1);
  args::ValueFlag<Index> gap(parser, "DEAD-TIME GAP", "Set gap value (default 2)", {'g', "gap"}, 2);
  args::ValueFlag<Index> src(
    parser, "SOURCES", "Number of ZINFANDEL sources (default 4)", {"src"}, 4);
  args::ValueFlag<Index> spokes(
    parser, "CAL SPOKES", "Number of spokes to use for calibration (default 5)", {"spokes"}, 5);
  args::ValueFlag<Index> read(
    parser, "CAL READ", "Read calibration size (default all)", {"read"}, 0);
  args::ValueFlag<float> l1(
    parser, "LAMBDA", "Tikhonov regularization (default 0)", {"lambda"}, 0.f);
  ParseCommand(parser, iname);

  HD5::RieslingReader reader(iname.Get());
  auto const traj = reader.trajectory();
  auto info = traj.info();
  auto out_info = info;
  if (volume) {
    out_info.volumes = 1;
  }

  HD5::Writer writer(OutName(iname.Get(), oname.Get(), "zinfandel", "h5"));
  writer.writeTrajectory(Trajectory(out_info, traj.points()));
  writer.writeMeta(reader.readMeta());

  Cx4 rad_ks = info.noncartesianSeries();
  for (Index iv = 0; iv < info.volumes; iv++) {
    Cx3 vol = reader.noncartesian(iv);
    zinfandel(gap.Get(), src.Get(), spokes.Get(), read.Get(), l1.Get(), traj.points(), vol);
    rad_ks.chip<3>(iv) = vol;
  }
  writer.writeTensor(rad_ks, HD5::Keys::Noncartesian);
  Log::Print(FMT_STRING("Finished"));
  return EXIT_SUCCESS;
}
