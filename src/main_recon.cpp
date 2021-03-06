#include "types.h"

#include "apodizer.h"
#include "cropper.h"
#include "fft_plan.h"
#include "filter.h"
#include "gridder.h"
#include "io_hd5.h"
#include "io_nifti.h"
#include "log.h"
#include "parse_args.h"
#include "sense.h"

int main_recon(args::Subparser &parser)
{
  COMMON_RECON_ARGS;

  args::Flag rss(parser, "RSS", "Use Root-Sum-Squares channel combination", {"rss", 'r'});
  args::Flag save_channels(
      parser, "CHANNELS", "Write out individual channels from first volume", {"channels", 'c'});

  Log log = ParseCommand(parser, iname);
  FFT::Start(log);
  HD5::Reader reader(iname.Get(), log);
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
  Cx4 grid = gridder.newMultichannel(info.channels);
  Cx3 image = cropper.newImage();
  Cx4 out = cropper.newSeries(info.volumes);
  out.setZero();
  image.setZero();
  FFT::ThreeDMulti fft(grid, log);

  long currentVolume = -1;
  Cx4 sense = cropper.newMultichannel(info.channels);
  if (senseFile) {
    sense = LoadSENSE(senseFile.Get(), cropper.dims(info.channels), log);
  } else {
    currentVolume = LastOrVal(senseVolume, info.volumes);
    reader.readNoncartesian(currentVolume, rad_ks);
    sense = DirectSENSE(traj, osamp.Get(), kernel, out_fov.Get(), rad_ks, senseLambda.Get(), log);
  }

  auto const &all_start = log.now();
  for (long iv = 0; iv < info.volumes; iv++) {
    auto const &vol_start = log.now();
    reader.readNoncartesian(iv, rad_ks);
    grid.setZero();
    gridder.toCartesian(rad_ks, grid);
    fft.reverse(grid);
    if (rss) {
      image.device(Threads::GlobalDevice()) =
          (cropper.crop4(grid) * cropper.crop4(grid).conjugate()).sum(Sz1{0}).sqrt();
    } else {
      image.device(Threads::GlobalDevice()) = (cropper.crop4(grid) * sense.conjugate()).sum(Sz1{0});
    }
    apodizer.deapodize(image);
    if (tukey_s || tukey_e || tukey_h) {
      ImageTukey(tukey_s.Get(), tukey_e.Get(), tukey_h.Get(), image, log);
    }
    out.chip(iv, 3) = image;
    log.info("Volume {}: {}", iv, log.toNow(vol_start));
    if (save_channels && (iv == 0)) {
      Cx4 const cropped = SwapToChannelLast(cropper.crop4(grid));
      WriteOutput(
          cropped, false, false, info, iname.Get(), oname.Get(), "channels", oftype.Get(), log);
    }
  }
  log.info("All volumes: {}", log.toNow(all_start));
  WriteOutput(out, mag, false, info, iname.Get(), oname.Get(), "recon", oftype.Get(), log);
  FFT::End(log);
  return EXIT_SUCCESS;
}
