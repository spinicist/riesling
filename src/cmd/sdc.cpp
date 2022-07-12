#include "types.h"

#include "io/hd5.hpp"
#include "log.h"
#include "parse_args.h"
#include "sdc.h"

using namespace rl;

int main_sdc(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "Input HD5 file");
  args::ValueFlag<std::string> oname(parser, "OUTPUT", "Override output name", {'o', "out"});
  args::ValueFlag<std::string> sdcType(
    parser, "SDC", "SDC type: 'pipe', 'pipenn' (default), 'radial'", {"sdc"}, "pipenn");
  args::ValueFlag<float> osamp(parser, "OS", "Oversampling when using pipenn", {'s', "os"}, 2.f);
  args::ValueFlag<Index> lores(parser, "L", "Number of lo-res spokes for radial", {'l', "lores"}, 0);
  args::ValueFlag<Index> gap(parser, "G", "Read-gap for radial", {'g', "gap"}, 0);
  args::ValueFlag<Index> its(parser, "N", "Maximum number of iterations (40)", {"max_its", 'n'}, 40);
  ParseCommand(parser, iname);
  HD5::RieslingReader reader(iname.Get());
  auto const traj = reader.trajectory();

  R2 dc;
  if (sdcType.Get() == "pipe") {
    dc = SDC::Pipe(traj, false, 2.1f, its.Get());
  } else if (sdcType.Get() == "pipenn") {
    dc = SDC::Pipe(traj, true, osamp.Get(), its.Get());
  } else if (sdcType.Get() == "radial") {
    dc = SDC::Radial(traj, lores.Get(), gap.Get());
  } else {
    Log::Fail(FMT_STRING("Uknown SDC method: {}"), sdcType.Get());
  }
  HD5::Writer writer(OutName(iname.Get(), oname.Get(), "sdc", "h5"));
  writer.writeTrajectory(traj);
  writer.writeTensor(dc, HD5::Keys::SDC);
  return EXIT_SUCCESS;
}
