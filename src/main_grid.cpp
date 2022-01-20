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
  args::Flag forward(parser, "F", "Apply forward gridding (to non-cartesian)", {'f', "fwd"});
  ParseCommand(parser, iname);
  HD5::Reader reader(iname.Get());
  auto const traj = reader.readTrajectory();
  auto const info = traj.info();

  auto const kernel = make_kernel(ktype.Get(), info.type, osamp.Get());
  auto const mapping = traj.mapping(kernel->inPlane(), osamp.Get());
  auto gridder = make_grid(kernel.get(), mapping, fastgrid);
  gridder->setSDC(SDC::Choose(sdc.Get(), traj, osamp.Get()));
  gridder->setSDCPower(sdcPow.Get());
  Cx3 rad_ks = info.noncartesianVolume();
  Cx5 grid(gridder->inputDimensions(info.channels));
  HD5::Writer writer(OutName(iname.Get(), oname.Get(), "grid", "h5"));
  writer.writeTrajectory(traj);
  auto const start = Log::Now();
  if (forward) {
    reader.readCartesian(grid);
    gridder->A(grid, rad_ks);
    writer.writeNoncartesian(
      rad_ks.reshape(Sz4{rad_ks.dimension(0), rad_ks.dimension(1), rad_ks.dimension(2), 1}));
    Log::Print("Wrote non-cartesian k-space. Took {}", Log::ToNow(start));
  } else {
    rad_ks = reader.noncartesian(0);
    gridder->Adj(rad_ks, grid);
    writer.writeTensor(grid, "cartesian");
    Log::Print("Wrote cartesian k-space. Took {}", Log::ToNow(start));
  }

  return EXIT_SUCCESS;
}
