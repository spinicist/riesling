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
  args::Positional<std::string> fname(parser, "FILE", "HD5 file to calculate trajectory from");
  args::ValueFlag<std::string> oname(parser, "OUTPUT", "Override output name", {"out", 'o'});
  args::ValueFlag<float> osamp(
      parser, "GRID OVERSAMPLE", "Oversampling factor for gridding, default 2", {'g', "grid"}, 2.f);
  args::MapFlag<std::string, SDC> sdc(
      parser, "SDC", "SDC Method. 0 - None, 1 - Analytic, 2 - Pipe", {"sdc"}, SDCMap);
  args::Flag kb(parser, "KB", "Use Kaiser-Bessel interpolation", {"kb"});
  args::ValueFlag<long> kw(
      parser, "KERNEL WIDTH", "Width of gridding kernel. Default 1 for NN, 3 for KB", {"kw"}, 3);
  Log log = ParseCommand(parser, fname);
  FFT::Start(log);
  HD5::Reader reader(fname.Get(), log);
  auto const &info = reader.info();
  Kernel *kernel =
      kb ? (Kernel *)new KaiserBessel(kw.Get(), osamp.Get(), (info.type == Info::Type::ThreeD))
         : (Kernel *)new NearestNeighbour(kw ? kw.Get() : 1);
  Gridder gridder(info, reader.readTrajectory(), osamp.Get(), sdc.Get(), kernel, log);
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
