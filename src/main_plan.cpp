#include "types.h"

#include "fft_plan.h"
#include "io_hd5.h"
#include "log.h"
#include "op/grid.h"
#include "op/grid-basis.h"
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
  auto gridder = make_grid(traj, osamp.Get(), kb, fastgrid, log);
  Cx4 grid4 = gridder->newMultichannel(traj.info().channels);
  Cx4 grid3 = gridder->newMultichannel(1);
  FFT::ThreeDMulti fft3(grid3, log);
  FFT::ThreeDMulti fft4(grid4, log);

  if (basisFile) {
    HD5::Reader basisReader(basisFile.Get(), log);
    R2 basis = basisReader.readBasis();
    long const nB = basis.dimension(1);
    // if ((traj.info().spokes_total() % basis.dimension(0)) != 0) {
    //   Log::Fail(
    //       FMT_STRING("Basis length {} does not evenly divide number of spokes {}"),
    //       basis.dimension(0),
    //       traj.info().spokes_total());
    // }
    auto gridderBasis = make_grid_basis(traj, osamp.Get(), kb, fastgrid, basis, log);
    auto const gridSz = gridderBasis->gridDims();
    Cx5 grid5(traj.info().channels, nB, gridSz[0], gridSz[1], gridSz[2]);
    FFT::ThreeDBasis fft(grid5, log);
  }

  FFT::End(log);
  return EXIT_SUCCESS;
}
