#include "types.h"

#include "fft_plan.h"
#include "gridder.h"
#include "io_hd5.h"
#include "kernels.h"
#include "log.h"
#include "parse_args.h"

int main_plan(args::Subparser &parser)
{
  COMMON_RECON_ARGS;
  args::ValueFlag<double> timelimit(
      parser, "LIMIT", "Time limit for FFT planning (default 60 s)", {"time", 't'}, 60.0);

  Log log = ParseCommand(parser, iname);
  FFT::Start(log);
  FFT::SetTimelimit(timelimit.Get());
  HD5::Reader reader(iname.Get(), log);
  auto const traj = reader.readTrajectory();
  Kernel *kernel =
      kb ? (Kernel *)new KaiserBessel(3, osamp.Get(), (traj.info().type == Info::Type::ThreeD))
         : (Kernel *)new NearestNeighbour();
  Gridder gridder(traj, osamp.Get(), kernel, fastgrid, log);
  Cx4 grid4 = gridder.newMultichannel(traj.info().channels);
  Cx4 grid3 = gridder.newMultichannel(1);
  FFT::ThreeDMulti fft3(grid3, log);
  FFT::ThreeDMulti fft4(grid4, log);
  FFT::End(log);
  return EXIT_SUCCESS;
}
