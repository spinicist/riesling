#include "types.h"

#include "fft_plan.h"
#include "io.h"
#include "log.h"
#include "op/grid-basis.h"
#include "op/grid.h"
#include "parse_args.h"
#include "tensorOps.h"
#include "threads.h"
#include <complex>

int main_traj(args::Subparser &parser)
{
  CORE_RECON_ARGS;

  args::ValueFlag<std::string> basisFile(
    parser, "BASIS", "Read subspace basis from .h5 file", {"basis", 'b'});
  args::Flag savePSF(parser, "PSF", "Write out Point-Spread-Function", {"psf", 'p'});

  Log log = ParseCommand(parser, iname);
  FFT::Start(log);
  HD5::Reader reader(iname.Get(), log);
  auto const traj = reader.readTrajectory();
  auto const info = traj.info();
  Cx3 rad_ks(1, info.read_points, info.spokes_total());
  rad_ks.setConstant(1.0f);

  auto gridder = make_grid(traj, osamp.Get(), kernel.Get(), fastgrid, log);
  gridder->setSDC(SDC::Choose(sdc.Get(), traj, osamp.Get(), log));
  Cx5 gridded;
  if (basisFile) {
    HD5::Reader basisReader(basisFile.Get(), log);
    R2 basis = basisReader.readBasis();
    Index const nB = basis.dimension(1);
    auto gridderBasis = make_grid_basis(gridder->mapping(), kernel.Get(), fastgrid, basis, log);
    gridderBasis->setSDC(gridder->SDC());
    auto const gridSz = gridderBasis->mapping().cartDims;
    gridded.resize(1, nB, gridSz[0], gridSz[1], gridSz[2]);
    gridderBasis->Adj(rad_ks, gridded);
  } else {
    gridded.resize(gridder->inputDimensions(1, info.echoes));
    gridder->Adj(rad_ks, gridded);
  }

  Cx5 psf;
  if (savePSF) {
    log.info("Calculating PSF");
    FFT::Planned<5, 3> fft(gridded.dimensions(), log);
    psf = gridded;
    fft.reverse(psf);
    psf = psf.shuffle(Sz5{1, 2, 3, 4, 0});
  }
  // Cheap hack to get equivalent dimensions to reconstructed images
  gridded = gridded.shuffle(Sz5{1, 2, 3, 4, 0});

  auto const fname = OutName(iname.Get(), oname.Get(), "traj", "h5");
  HD5::Writer writer(fname, log);
  writer.writeTensor(gridded, "traj-image");
  if (savePSF) {
    writer.writeTensor(psf, "psf-image");
  }

  FFT::End(log);
  return EXIT_SUCCESS;
}
