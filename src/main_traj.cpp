#include "types.h"

#include "fft_plan.h"
#include "io.h"
#include "log.h"
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
  Cx3 rad_ks(1, info.read_points, info.spokes);
  rad_ks.setConstant(1.0f);

  Cx4 out;
  if (basisFile) {
    HD5::Reader basisReader(basisFile.Get(), log);
    R2 basis = basisReader.readBasis();
    auto gridder = make_grid_basis(traj, osamp.Get(), kernel.Get(), fastgrid, basis, log);
    gridder->setSDC(SDC::Choose(sdc.Get(), traj, osamp.Get(), log));
    gridder->setSDCPower(sdcPow.Get());
    Cx5 grid(gridder->inputDimensions(1));
    gridder->Adj(rad_ks, grid);
    out = grid.chip<0>(0);
  } else {
    auto gridder = make_grid(traj, osamp.Get(), kernel.Get(), fastgrid, log);
    gridder->setSDC(SDC::Choose(sdc.Get(), traj, osamp.Get(), log));
    gridder->setSDCPower(sdcPow.Get());
    Cx5 grid(gridder->inputDimensions(1));
    gridder->Adj(rad_ks, grid);
    out = grid.chip<0>(0);
  }

  auto const fname = OutName(iname.Get(), oname.Get(), "traj", "h5");
  HD5::Writer writer(fname, log);
  writer.writeTensor(
    Cx5(
      out.reshape(Sz5{out.dimension(0), out.dimension(1), out.dimension(2), out.dimension(3), 1})),
    "traj-image");

  if (savePSF) {
    log.info("Calculating PSF");
    FFT::Planned<4, 3> fft(out.dimensions(), log);
    fft.reverse(out);
    writer.writeTensor(
      Cx5(out.reshape(
        Sz5{out.dimension(0), out.dimension(1), out.dimension(2), out.dimension(3), 1})),
      "psf-image");
  }

  FFT::End(log);
  return EXIT_SUCCESS;
}
