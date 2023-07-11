#include "types.hpp"

#include "fft/fft.hpp"
#include "io/hd5.hpp"
#include "log.hpp"
#include "op/make_grid.hpp"
#include "parse_args.hpp"

using namespace rl;

int main_plan(args::Subparser &parser)
{
  CoreOpts                coreOpts(parser);
  args::ValueFlag<double> timelimit(parser, "L", "Time limit for FFT planning (default 60 s)", {"time", 't'}, 60.0);
  args::ValueFlag<Index>  channels(parser, "N", "Number of channels to plan (default 1)", {"channels", 'c'}, 1);
  ParseCommand(parser, coreOpts.iname);

  FFT::SetTimelimit(timelimit.Get());
  HD5::Reader reader(coreOpts.iname.Get());
  Trajectory  traj(reader.readInfo(), reader.readTensor<Re3>(HD5::Keys::Trajectory));
  auto const  basis = ReadBasis(coreOpts.basisFile.Get());
  auto        gridder = make_grid<Cx, 3>(traj, coreOpts.ktype.Get(), coreOpts.osamp.Get(), channels.Get(), basis);
  auto const  fftN = FFT::Make<5, 3>(gridder->ishape);
  return EXIT_SUCCESS;
}
