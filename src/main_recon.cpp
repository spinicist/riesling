#include "types.h"

#include "cropper.h"
#include "fft_plan.h"
#include "filter.h"
#include "io_hd5.h"
#include "io_nifti.h"
#include "log.h"
#include "op/grid.h"
#include "parse_args.h"
#include "sense.h"
#include "tensorOps.h"

int main_recon(args::Subparser &parser)
{
  COMMON_RECON_ARGS;
  COMMON_SENSE_ARGS;
  args::Flag rss(parser, "RSS", "Use Root-Sum-Squares channel combination", {"rss", 'r'});
  args::Flag save_channels(
      parser, "CHANNELS", "Write out individual channels from first volume", {"channels", 'c'});

  Log log = ParseCommand(parser, iname);
  FFT::Start(log);
  HD5::Reader reader(iname.Get(), log);
  auto const traj = reader.readTrajectory();
  auto const &info = traj.info();
  auto gridder = make_grid(traj, osamp.Get(), kb, fastgrid, log);
  SDC::Choose(sdc.Get(), traj, gridder, log);
  gridder->setSDCExponent(sdc_exp.Get());
  Cropper cropper(info, gridder->gridDims(), out_fov.Get(), log);
  R3 const apo = gridder->apodization(cropper.size());
  Cx3 rad_ks = info.noncartesianVolume();
  Cx4 grid = gridder->newMultichannel(info.channels);
  Cx3 image = cropper.newImage();
  Cx4 out = cropper.newSeries(info.volumes);
  out.setZero();
  image.setZero();
  FFT::ThreeDMulti fft(grid, log);

  long currentVolume = -1;
  Cx4 sense = rss ? Cx4() : cropper.newMultichannel(info.channels);
  if (!rss) {
    if (senseFile) {
      sense = LoadSENSE(senseFile.Get(), log);
    } else {
      currentVolume = LastOrVal(senseVolume, info.volumes);
      reader.readNoncartesian(currentVolume, rad_ks);
      sense = DirectSENSE(traj, osamp.Get(), kb, out_fov.Get(), rad_ks, senseLambda.Get(), log);
    }
  }

  auto dev = Threads::GlobalDevice();
  auto const &all_start = log.now();
  for (long iv = 0; iv < info.volumes; iv++) {
    auto const &vol_start = log.now();
    reader.readNoncartesian(iv, rad_ks);
    grid.setZero();
    gridder->Adj(rad_ks, grid);
    log.info("FFT...");
    fft.reverse(grid);
    log.info("Channel combination...");
    if (rss) {
      image.device(dev) = ConjugateSum(cropper.crop4(grid), cropper.crop4(grid)).sqrt();
    } else {
      image.device(dev) = ConjugateSum(cropper.crop4(grid), sense);
    }
    image.device(dev) = image / apo.cast<Cx>();
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
