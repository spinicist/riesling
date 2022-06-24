#include "types.h"

#include "io/hd5.hpp"
#include "log.h"
#include "op/grids.h"
#include "parse_args.h"
#include "sdc.h"
#include "threads.h"
#include <filesystem>

int main_grid(args::Subparser &parser)
{
  CoreOpts core(parser);
  SDC::Opts sdcOpts(parser);
  args::Flag fwd(parser, "", "Apply forward operation", {'f', "fwd"});
  ParseCommand(parser, core.iname);
  HD5::RieslingReader reader(core.iname.Get());
  auto const traj = reader.trajectory();
  auto const info = traj.info();

  auto const kernel = make_kernel(core.ktype.Get(), info.type, core.osamp.Get());
  auto const mapping = traj.mapping(kernel->inPlane(), core.osamp.Get());
  auto gridder = make_grid(kernel.get(), mapping, info.channels, core.fast);
  Cx3 rad_ks = info.noncartesianVolume();
  HD5::Writer writer(OutName(core.iname.Get(), core.oname.Get(), "grid", "h5"));
  writer.writeTrajectory(traj);
  auto const start = Log::Now();
  if (fwd) {
    rad_ks = gridder->A(reader.readTensor<Cx5>(HD5::Keys::Cartesian));
    writer.writeTensor(
      Cx4(rad_ks.reshape(Sz4{rad_ks.dimension(0), rad_ks.dimension(1), rad_ks.dimension(2), 1})),
      HD5::Keys::Noncartesian);
    Log::Print(FMT_STRING("Wrote non-cartesian k-space. Took {}"), Log::ToNow(start));
  } else {
    auto const sdc = SDC::Choose(sdcOpts, traj, core.osamp.Get());
    writer.writeTensor(gridder->Adj(sdc->Adj(reader.noncartesian(0))), HD5::Keys::Cartesian);
    Log::Print(FMT_STRING("Wrote cartesian k-space. Took {}"), Log::ToNow(start));
  }

  return EXIT_SUCCESS;
}
