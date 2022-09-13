#include "types.h"

#include "io/hd5.hpp"
#include "log.hpp"
#include "parse_args.hpp"
#include "sdc.h"

using namespace rl;

int main_sdc(args::Subparser &parser)
{
  CoreOpts core(parser);
  args::ValueFlag<std::string> sdcType(
    parser, "SDC", "SDC type: 'pipe', 'radial'", {"sdc"}, "pipe");
  args::ValueFlag<Index> lores(parser, "L", "Number of lo-res traces for radial", {'l', "lores"}, 0);
  args::ValueFlag<Index> gap(parser, "G", "Read-gap for radial", {'g', "gap"}, 0);
  args::ValueFlag<Index> its(parser, "N", "Maximum number of iterations (40)", {"max-its", 'n'}, 40);
  ParseCommand(parser, core.iname);
  HD5::RieslingReader reader(core.iname.Get());
  auto const traj = reader.trajectory();

  Re2 dc;
  if (sdcType.Get() == "pipe") {
    dc = SDC::Pipe(traj, core.ktype.Get(), core.osamp.Get(), its.Get());
  } else if (sdcType.Get() == "pipenn") {
    dc = SDC::Pipe(traj, core.ktype.Get(), core.osamp.Get(), its.Get());
  } else if (sdcType.Get() == "radial") {
    dc = SDC::Radial(traj, lores.Get(), gap.Get());
  } else {
    Log::Fail(FMT_STRING("Uknown SDC method: {}"), sdcType.Get());
  }
  HD5::Writer writer(OutName(core.iname.Get(), core.oname.Get(), "sdc", "h5"));
  writer.writeTrajectory(traj);
  writer.writeTensor(dc, HD5::Keys::SDC);
  return EXIT_SUCCESS;
}
