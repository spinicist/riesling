#include "types.h"

#include "fft3.h"
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
  args::ValueFlag<std::string> oname(parser, "OUTPUT", "Override output name", {"out", 'o'});
  args::ValueFlag<std::string> sdc(
      parser, "SDC FILE", "Load SDC from this h5 file", {"sdc"}, "pipe");
  Log log = ParseCommand(parser, fname);
  FFT::Start(log);
  HD5::Reader reader(fname.Get(), log);
  auto const traj = reader.readTrajectory();
  auto const info = traj.info();

  Kernel *kernel =
      kb ? (Kernel *)new KaiserBessel(kw.Get(), osamp.Get(), (info.type == Info::Type::ThreeD))
         : (Kernel *)new NearestNeighbour(kw ? kw.Get() : 1);
  Gridder gridder(traj, osamp.Get(), kernel, fastgrid, log);
  SDC::Load(sdc.Get(), traj, gridder, log);
  Cx3 grid = gridder.newGrid1();
  FFT3 fft(grid, log);

  grid.setZero();
  Cx2 rad_ks(info.read_points, info.spokes_total());
  rad_ks.setConstant(1.0f);
  gridder.toCartesian(rad_ks, grid);
  WriteNifti(info, R3(grid.abs()), OutName(fname, oname, "traj"), log);
  fft.reverse(grid);
  WriteNifti(info, grid, OutName(fname, oname, "psf"), log);
  FFT::End(log);
  return EXIT_SUCCESS;
}
