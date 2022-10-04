#include "types.hpp"

#include "fft/fft.hpp"
#include "io/hd5.hpp"
#include "log.hpp"
#include "op/gridBase.hpp"
#include "parse_args.hpp"

using namespace rl;

int main_plan(args::Subparser &parser)
{
  CoreOpts core(parser);
  ExtraOpts extra(parser);
  args::ValueFlag<double> timelimit(parser, "L", "Time limit for FFT planning (default 60 s)", {"time", 't'}, 60.0);
  args::ValueFlag<Index> channels(parser, "N", "Number of channels to plan (default 1)", {"channels", 'c'}, 1);
  ParseCommand(parser, core.iname);

  FFT::SetTimelimit(timelimit.Get());
  HD5::Reader reader(core.iname.Get());
  Trajectory traj(reader);
  auto const basis = ReadBasis(core.basisFile);
  auto gridder = make_grid<Cx, 3>(traj, core.ktype.Get(), core.osamp.Get(), channels.Get(), basis);
  auto const fftN = FFT::Make<5, 3>(gridder->inputDimensions());
  return EXIT_SUCCESS;
}
