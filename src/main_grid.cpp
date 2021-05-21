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
  HD5::Reader reader(fname.Get(), log);
  auto const traj = reader.readTrajectory();
  auto const info = reader.info();

  Kernel *kernel =
      kb ? (Kernel *)new KaiserBessel(kw.Get(), osamp.Get(), (info.type == Info::Type::ThreeD))
         : (Kernel *)new NearestNeighbour(kw ? kw.Get() : 1);
  Gridder gridder(traj, osamp.Get(), kernel, fastgrid, log);
  SDC::Load(sdc.Get(), traj, gridder, log);
  gridder.setSDCExponent(sdc_exp.Get());
  Cx3 rad_ks = info.noncartesianVolume();
  Cx4 grid = gridder.newGrid();

  long const vol = volume ? volume.Get() : 0;
  auto const &vol_start = log.now();

  HD5::Writer writer(OutName(fname, oname, "grid", "h5"), log);
  writer.writeInfo(info);
  writer.writeTrajectory(traj);
  if (forward) {
    reader.readCartesian(grid);
    gridder.toNoncartesian(grid, rad_ks);
    writer.writeNoncartesian(
        rad_ks.reshape(Sz4{rad_ks.dimension(0), rad_ks.dimension(1), rad_ks.dimension(2), 1}));
    log.info("Wrote non-cartesian k-space. Took {}", log.toNow(vol_start));
  } else {
    reader.readNoncartesian(vol, rad_ks);
    gridder.toCartesian(rad_ks, grid);
    writer.writeCartesian(grid);
    log.info("Wrote cartesian k-space. Took {}", log.toNow(vol_start));
  }

  return EXIT_SUCCESS;
}
