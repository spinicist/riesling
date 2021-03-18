#include "types.h"

#include "io_hd5.h"
#include "log.h"
#include "parse_args.h"

int main_split(args::Subparser &parser)
{
  COMMON_RECON_ARGS;

  args::ValueFlag<long> nspokes(parser, "NUM SPOKES", "Spokes per volume", {"n", "nspokes"}, -1);
  args::ValueFlag<long> vol(parser, "VOLUME", "Only take this volume", {"v", "vol"}, 0);
  args::ValueFlag<float> ds(parser, "DS", "Downsample by factor", {"ds"}, 1.0);

  Log log = ParseCommand(parser, fname);

  HD5Reader reader(fname.Get(), log);
  auto const &info = reader.info();
  auto const trajectory = reader.readTrajectory();

  Cx3 rad_ks = info.noncartesianVolume();
  reader.readData(vol.Get(), rad_ks);

  return EXIT_SUCCESS;
}