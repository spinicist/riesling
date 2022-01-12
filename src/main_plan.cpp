#include "types.h"

#include "fft_plan.h"
#include "io.h"
#include "log.h"
#include "op/grid.h"
#include "parse_args.h"

int main_plan(args::Subparser &parser)
{
  COMMON_RECON_ARGS;
  args::ValueFlag<double> timelimit(
    parser, "LIMIT", "Time limit for FFT planning (default 60 s)", {"time", 't'}, 60.0);
  args::ValueFlag<std::string> basisFile(
    parser, "BASIS", "Read subspace basis from .h5 file", {"basis", 'b'});

  Log log = ParseCommand(parser, iname);
  FFT::Start(log);
  FFT::SetTimelimit(timelimit.Get());
  HD5::Reader reader(iname.Get(), log);
  auto const traj = reader.readTrajectory();
  auto gridder = make_grid(traj, osamp.Get(), kernel.Get(), fastgrid, log);
  FFT::Planned<5, 3> fft3(gridder->inputDimensions(traj.info().channels), log);
  FFT::Planned<5, 3> fft4(gridder->inputDimensions(1), log);

  if (basisFile) {
    HD5::Reader basisReader(basisFile.Get(), log);
    R2 basis = basisReader.readBasis();
    auto gridderBasis = make_grid_basis(traj, osamp.Get(), kernel.Get(), fastgrid, basis, log);
    FFT::Planned<5, 3> fft(gridderBasis->inputDimensions(traj.info().channels), log);
  }

  FFT::End(log);
  return EXIT_SUCCESS;
}
