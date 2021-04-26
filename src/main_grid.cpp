#include "types.h"

#include "gridder.h"
#include "io_hd5.h"
#include "io_nifti.h"
#include "log.h"
#include "parse_args.h"
#include "threads.h"
#include <filesystem>

int main_grid(args::Subparser &parser)
{
  COMMON_RECON_ARGS;
  args::Flag forward(parser, "REV", "Apply forward gridding (to non-cartesian)", {'f', "fwd"});
  Log log = ParseCommand(parser, fname);
  HD5Reader reader(fname.Get(), log);
  auto const &info = reader.info();
  auto const &traj = reader.readTrajectory();
  Kernel *kernel =
      kb ? (Kernel *)new KaiserBessel(kw.Get(), osamp.Get(), (info.type == Info::Type::ThreeD))
         : (Kernel *)new NearestNeighbour(kw ? kw.Get() : 1);
  Gridder gridder(info, traj, osamp.Get(), sdc.Get(), kernel, log);
  gridder.setDCExponent(dc_exp.Get());
  Cx3 rad_ks = info.noncartesianVolume();
  Cx4 grid = gridder.newGrid();

  long const vol = volume ? volume.Get() : 0;
  auto const &vol_start = log.now();

  HD5Writer writer(OutName(fname, oname, "grid", "h5"), log);
  writer.writeInfo(info);
  writer.writeTrajectory(traj);
  if (forward) {
    reader.readData(grid, "grid");
    gridder.toNoncartesian(grid, rad_ks);
    writer.writeVolume(0, rad_ks);
    log.info("Wrote non-cartesian k-space. Took {}", log.toNow(vol_start));
  } else {
    reader.readVolume(vol, rad_ks);
    gridder.toCartesian(rad_ks, grid);
    writer.writeData(grid, "grid");
    log.info("Wrote cartesian k-space. Took {}", log.toNow(vol_start));
  }

  return EXIT_SUCCESS;
}
