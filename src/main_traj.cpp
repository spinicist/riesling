#include "types.h"

#include "fft_plan.h"
#include "gridder.h"
#include "io_hd5.h"
#include "io_nifti.h"
#include "log.h"
#include "parse_args.h"
#include "threads.h"
#include <complex>

int main_traj(args::Subparser &parser)
{
  CORE_RECON_ARGS;
  Log log = ParseCommand(parser, iname);
  FFT::Start(log);
  HD5::Reader reader(iname.Get(), log);
  auto const traj = reader.readTrajectory();
  auto const info = traj.info();

  Kernel *kernel =
      kb ? (Kernel *)new KaiserBessel(kw.Get(), osamp.Get(), (info.type == Info::Type::ThreeD))
         : (Kernel *)new NearestNeighbour(kw ? kw.Get() : 1);
  Gridder gridder(traj, osamp.Get(), kernel, fastgrid, log);
  SDC::Load(sdc.Get(), traj, gridder, log);
  Cx4 grid = gridder.newMultichannel(1);
  FFT::ThreeDMulti fft(grid, log);

  grid.setZero();
  Cx3 rad_ks(1, info.read_points, info.spokes_total());
  rad_ks.setConstant(1.0f);
  gridder.toCartesian(rad_ks, grid);
  Cx4 output = SwapToChannelLast(grid);
  WriteOutput(output, true, false, info, iname.Get(), oname.Get(), "traj", oftype.Get(), log);
  fft.reverse(grid);
  output = SwapToChannelLast(grid);
  WriteOutput(output, false, false, info, iname.Get(), oname.Get(), "psf", oftype.Get(), log);
  FFT::End(log);
  return EXIT_SUCCESS;
}
