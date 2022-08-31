#include "types.h"

#include "fft/fft.hpp"
#include "io/hd5.hpp"
#include "log.h"
#include "op/gridBase.hpp"
#include "parse_args.h"

using namespace rl;

int main_plan(args::Subparser &parser)
{
  CoreOpts core(parser);
  ExtraOpts extra(parser);
  args::ValueFlag<double> timelimit(parser, "LIMIT", "Time limit for FFT planning (default 60 s)", {"time", 't'}, 60.0);

  ParseCommand(parser, core.iname);

  FFT::SetTimelimit(timelimit.Get());
  HD5::RieslingReader reader(core.iname.Get());
  auto const traj = reader.trajectory();
  auto const info = traj.info();
  auto const basis = ReadBasis(core.basisFile);
  auto gridder = make_grid<Cx, 3>(traj, core.ktype.Get(), core.osamp.Get(), info.channels, basis);
  auto const fftN = FFT::Make<5, 3>(gridder->inputDimensions());
  auto grid1 = make_grid<Cx, 3>(traj, core.ktype.Get(), core.osamp.Get(), 1, basis);
  auto const fft1 = FFT::Make<5, 3>(grid1->inputDimensions());
  return EXIT_SUCCESS;
}
