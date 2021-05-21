#include "types.h"

#include "apodizer.h"
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
  HD5::Reader reader(fname.Get(), log);
  auto const traj = reader.readTrajectory();
  auto const &info = traj.info();
  Kernel *kernel =
      kb ? (Kernel *)new KaiserBessel(kw.Get(), osamp.Get(), (info.type == Info::Type::ThreeD))
         : (Kernel *)new NearestNeighbour(kw ? kw.Get() : 1);
  Gridder gridder(traj, osamp.Get(), kernel, fastgrid, log);
  SDC::Load(sdc.Get(), traj, gridder, log);
  gridder.setSDCExponent(sdc_exp.Get());
  Cropper cropper(info, gridder.gridDims(), out_fov.Get(), log);
  Apodizer apodizer(kernel, gridder.gridDims(), cropper.size(), log);
  Cx3 rad_ks = info.noncartesianVolume();
  Cx4 grid = gridder.newGrid();
  Cx3 image = cropper.newImage();
  R4 out = cropper.newRealSeries(info.volumes);
  out.setZero();
  image.setZero();

  FFT3N fft(grid, log);

  auto const &all_start = log.now();
  for (auto const &iv : WhichVolumes(volume.Get(), info.volumes)) {
    auto const &vol_start = log.now();
    reader.readNoncartesian(iv, rad_ks);
    grid.setZero();
    gridder.toCartesian(rad_ks, grid);
    fft.reverse();
    image.device(Threads::GlobalDevice()) =
        (cropper.crop4(grid) * cropper.crop4(grid).conjugate()).sum(Sz1{0}).sqrt();
    apodizer.deapodize(image);
    if (tukey_s || tukey_e || tukey_h) {
      ImageTukey(tukey_s.Get(), tukey_e.Get(), tukey_h.Get(), image, log);
    }
    out.chip(iv, 3) = image.real();
    log.info("Volume {}: {}", iv, log.toNow(vol_start));
  }
  log.info("All volumes: {}", log.toNow(all_start));

  WriteVolumes(info, out, volume.Get(), OutName(fname, oname, "rss", outftype.Get()), log);
  FFT::End(log);
  return EXIT_SUCCESS;
}
