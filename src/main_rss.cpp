#include "types.h"

#include "cropper.h"
#include "fft3n.h"
#include "filter.h"
#include "gridder.h"
#include "io_hd5.h"
#include "io_nifti.h"
#include "log.h"
#include "parse_args.h"

int main_rss(args::Subparser &parser)
{
  COMMON_RECON_ARGS;

  Log log = ParseCommand(parser, fname);
  FFT::Start(log);
  HD5Reader reader(fname.Get(), log);
  auto const &info = reader.info();
  Kernel *kernel =
      kb ? (Kernel *)new KaiserBessel(3, osamp.Get(), !stack) : (Kernel *)new NearestNeighbour();
  Gridder gridder(info, reader.readTrajectory(), osamp.Get(), est_dc, kernel, stack, log);
  gridder.setDCExponent(dc_exp.Get());
  Cropper cropper(info, gridder.gridDims(), out_fov.Get(), stack, log);
  Cx3 rad_ks = info.noncartesianVolume();
  Cx4 grid = gridder.newGrid();
  Cx3 image = cropper.newImage();
  R4 out = cropper.newRealSeries(info.volumes);
  out.setZero();
  image.setZero();

  FFT3N fft(grid, log);

  auto const &all_start = log.start_time();
  for (auto const &iv : WhichVolumes(volume.Get(), info.volumes)) {
    auto const &vol_start = log.start_time();
    reader.readData(iv, rad_ks);
    grid.setZero();
    gridder.toCartesian(rad_ks, grid);
    fft.reverse();
    image.device(Threads::GlobalDevice()) =
        (cropper.crop4(grid) * cropper.crop4(grid).conjugate()).sum(Sz1{0}).sqrt();
    gridder.deapodize(image);
    if (tukey_s || tukey_e || tukey_h) {
      ImageTukey(tukey_s.Get(), tukey_e.Get(), tukey_h.Get(), image, log);
    }
    out.chip(iv, 3) = image.real();
    log.stop_time(vol_start, "Volume took");
  }
  log.stop_time(all_start, "All volumes took");

  WriteVolumes(info, out, volume.Get(), OutName(fname, oname, "rss"), log);
  FFT::End(log);
  return EXIT_SUCCESS;
}
