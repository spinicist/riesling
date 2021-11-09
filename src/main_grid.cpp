#include "types.h"

#include "io.h"
#include "log.h"
#include "op/grid.h"
#include "parse_args.h"
#include "threads.h"
#include <filesystem>

int main_grid(args::Subparser &parser)
{
  COMMON_RECON_ARGS;
  args::Flag forward(parser, "REV", "Apply forward gridding (to non-cartesian)", {'f', "fwd"});
  Log log = ParseCommand(parser, iname);
  HD5::Reader reader(iname.Get(), log);
  auto const traj = reader.readTrajectory();
  auto const info = traj.info();

  auto gridder = make_grid(traj, osamp.Get(), kb, fastgrid, log);
  gridder->setSDC(SDC::Choose(sdc.Get(), traj, gridder, log));
  Cx3 rad_ks = info.noncartesianVolume();
  Cx4 grid = gridder->newMultichannel(info.channels);

  auto const &vol_start = log.now();

  HD5::Writer writer(OutName(iname.Get(), oname.Get(), "grid", "h5"), log);
  writer.writeTrajectory(traj);
  if (forward) {
    reader.readCartesian(grid);
    gridder->A(grid, rad_ks);
    writer.writeNoncartesian(
      rad_ks.reshape(Sz4{rad_ks.dimension(0), rad_ks.dimension(1), rad_ks.dimension(2), 1}));
    log.info("Wrote non-cartesian k-space. Took {}", log.toNow(vol_start));
  } else {
    rad_ks = reader.noncartesian(0);
    gridder->Adj(rad_ks, grid);
    writer.writeCartesian(grid);
    log.info("Wrote cartesian k-space. Took {}", log.toNow(vol_start));
  }

  return EXIT_SUCCESS;
}
