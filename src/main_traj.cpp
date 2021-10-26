#include "types.h"

#include "fft_plan.h"
#include "io_hd5.h"
#include "io_nifti.h"
#include "log.h"
#include "op/grid.h"
#include "op/grid-basis.h"
#include "parse_args.h"
#include "tensorOps.h"
#include "threads.h"
#include <complex>

int main_traj(args::Subparser &parser)
{
  CORE_RECON_ARGS;

  args::ValueFlag<std::string> basisFile(
      parser, "BASIS", "Read subspace basis from .h5 file", {"basis", 'b'});

  Log log = ParseCommand(parser, iname);
  FFT::Start(log);
  HD5::Reader reader(iname.Get(), log);
  auto const traj = reader.readTrajectory();
  auto const info = traj.info();
  Cx3 rad_ks(1, info.read_points, info.spokes_total());
  rad_ks.setConstant(1.0f);

  auto gridder = make_grid(traj, osamp.Get(), kb, fastgrid, log);
  SDC::Choose(sdc.Get(), traj, gridder, log);
  Cx4 gridded, psf;

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
    auto gridderBasis = make_grid_basis(gridder->mapping(), kb, fastgrid, basis, log);
    gridderBasis->setSDC(gridder->SDC());
    auto const gridSz = gridderBasis->gridDims();
    Cx5 grid5(1, nB, gridSz[0], gridSz[1], gridSz[2]);
    gridderBasis->Adj(rad_ks, grid5);
    Cx4 grid = gridder->newMultichannel(nB);
    FFT::ThreeDMulti fft(grid, log);
    grid = grid5.chip(0, 0);
    gridded = FirstToLast4(grid);
    fft.reverse(grid);
    psf = FirstToLast4(grid);
  } else {
    Cx4 grid = gridder->newMultichannel(1);
    FFT::ThreeDMulti fft(grid, log);
    gridder->Adj(rad_ks, grid);
    gridded = FirstToLast4(grid);
    fft.reverse(grid);
    psf = FirstToLast4(grid);
  }
  WriteOutput(gridded, false, false, info, iname.Get(), oname.Get(), "traj", oftype.Get(), log);
  WriteOutput(psf, false, false, info, iname.Get(), oname.Get(), "psf", oftype.Get(), log);
  FFT::End(log);
  return EXIT_SUCCESS;
}
