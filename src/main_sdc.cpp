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
  args::ValueFlag<std::string> oftype(
    parser, "OUT FILETYPE", "File type of output (nii/nii.gz/img/h5)", {"oft"}, "h5");
  args::ValueFlag<float> sdcPow(parser, "P", "SDC Power (default 1.0)", {'p', "pow"}, 1.0f);
  args::ValueFlag<float> osamp(parser, "OS", "Oversampling when using pipenn", {'s', "os"}, 2.f);
  Log log = ParseCommand(parser, iname);
  HD5::Reader reader(iname.Get(), log);
  auto const traj = reader.readTrajectory();
  auto const &info = traj.info();

  R2 dc;
  if (sdc.Get() == "pipe") {
    dc = SDC::Pipe(traj, false, 2.1f, log);
  } else if (sdc.Get() == "pipenn") {
    dc = SDC::Pipe(traj, true, osamp.Get(), log);
  } else if (sdc.Get() == "radial") {
    dc = SDC::Radial(traj, log);
  } else {
    Log::Fail(FMT_STRING("Uknown SDC method: {}"), sdc.Get());
  }
  if (sdcPow) {
    dc = dc.pow(sdcPow.Get());
  }
  HD5::Writer writer(OutName(iname.Get(), oname.Get(), "sdc", "h5"), log);
  writer.writeInfo(info);
  writer.writeSDC(dc);
  return EXIT_SUCCESS;
}
