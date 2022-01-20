#include "types.h"

#include "io.h"
#include "log.h"
#include "op/grid.h"
#include "parse_args.h"

int main_sdc(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "Input HD5 file");
  args::ValueFlag<std::string> oname(parser, "OUTPUT", "Override output name", {'o', "out"});
  args::ValueFlag<std::string> sdc(
    parser, "SDC", "SDC type: 'pipe', 'pipenn', 'radial'", {"sdc"}, "pipe");
  args::ValueFlag<float> osamp(parser, "OS", "Oversampling when using pipenn", {'s', "os"}, 2.f);
  args::ValueFlag<Index> lores(
    parser, "L", "Number of lo-res spokes for radial", {'l', "lores"}, 0);
  args::ValueFlag<Index> gap(parser, "G", "Read-gap for radial", {'g', "gap"}, 0);
  ParseCommand(parser, iname);
  HD5::Reader reader(iname.Get());
  auto const traj = reader.readTrajectory();
  auto const &info = traj.info();

  R2 dc;
  if (sdc.Get() == "pipe") {
    dc = SDC::Pipe(traj, false, 2.1f);
  } else if (sdc.Get() == "pipenn") {
    dc = SDC::Pipe(traj, true, osamp.Get());
  } else if (sdc.Get() == "radial") {
    dc = SDC::Radial(traj, lores.Get(), gap.Get());
  } else {
    Log::Fail(FMT_STRING("Uknown SDC method: {}"), sdc.Get());
  }
  HD5::Writer writer(OutName(iname.Get(), oname.Get(), "sdc", "h5"));
  writer.writeInfo(info);
  writer.writeSDC(dc);
  return EXIT_SUCCESS;
}
