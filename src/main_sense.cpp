#include "types.h"

#include "cropper.h"
#include "fft.h"
#include "filter.h"
#include "gridder.h"
#include "io_hd5.h"
#include "io_nifti.h"
#include "log.h"
#include "parse_args.h"
#include "sense.h"

int main_sense(args::Subparser &parser)
{
  COMMON_RECON_ARGS;

  args::Flag magnitude(parser, "MAGNITUDE", "Output magnitude images only", {"magnitude"});
  args::ValueFlag<long> sense_vol(
      parser, "SENSE VOLUME", "Take SENSE maps from this volume (default last)", {"sense_vol"}, -1);
  args::Flag save_maps(parser, "SAVE MAPS", "Write out sensitivity maps", {"maps", 'm'});
  args::Flag save_channels(
      parser, "SAVE CHANNELS", "Write out individual channel images", {"channels", 'c'});
  args::ValueFlag<long> esense(
      parser,
      "EIGENSENSE",
      "Use eigen-mode normalization with N channels (default 4)",
      {"esense"},
      4);

  Log log = ParseCommand(parser, fname);
  FFTStart(log);

  HD5Reader reader(fname.Get(), log);
  auto const &info = reader.info();
  auto const trajectory = reader.readTrajectory();

  Gridder gridder(info, trajectory, osamp.Get(), est_dc, kb, stack, log);
  gridder.setDCExponent(dc_exp.Get());

  Cropper cropper(info, gridder.gridDims(), out_fov.Get(), stack, log);
  Cx3 rad_ks = info.noncartesianVolume();
  long currentVolume = SenseVolume(sense_vol, info.volumes);
  reader.readData(currentVolume, rad_ks);
  Cx4 sense = esense ? cropper.crop4(EigenSENSE(
                           info, trajectory, osamp.Get(), stack, kb, esense.Get(), rad_ks, log))
                     : cropper.crop4(SENSE(info, trajectory, osamp.Get(), stack, kb, rad_ks, log));
  if (save_maps) {
    WriteNifti(info, Cx4(sense.shuffle(Sz4{1, 2, 3, 0})), OutName(fname, oname, "sense-maps"), log);
  }

  Cx4 grid = gridder.newGrid();
  grid.setZero();
  FFT3N fft(grid, log);
  Cx3 image = cropper.newImage();
  Cx4 out = cropper.newSeries(info.volumes);
  Cx4 channel_images = cropper.newMultichannel(info.channels);
  auto const &all_start = log.start_time();
  bool channels_saved = false;
  for (auto const &iv : WhichVolumes(volume.Get(), info.volumes)) {
    log.info(FMT_STRING("Processing volume: {}"), iv);
    auto const &vol_start = log.start_time();
    if (iv != currentVolume) { // For single volume images, we already read it for SENSE
      reader.readData(iv, rad_ks);
      currentVolume = iv;
    }
    grid.setZero();
    gridder.toCartesian(rad_ks, grid);
    fft.reverse();
    channel_images.device(Threads::GlobalDevice()) = cropper.crop4(grid) * sense.conjugate();
    image.device(Threads::GlobalDevice()) = channel_images.sum(Sz1{0});
    gridder.deapodize(image);
    if (tukey_s || tukey_e || tukey_h) {
      ImageTukey(tukey_s.Get(), tukey_e.Get(), tukey_h.Get(), image, log);
    }

    out.chip(iv, 3) = image;
    log.stop_time(vol_start, "Volume took");
    if (save_channels && !channels_saved) {
      WriteNifti(
          info,
          Cx4(channel_images.shuffle(Sz4{1, 2, 3, 0})),
          OutName(fname, oname, "sense-channels"),
          log);
      channels_saved = true;
    }
  }
  log.stop_time(all_start, "All volumes took");

  auto const ofile = OutName(fname, oname, "sense");
  if (magnitude) {
    WriteVolumes(info, R4(out.abs()), volume.Get(), ofile, log);
  } else {
    WriteVolumes(info, out, volume.Get(), ofile, log);
  }

  FFTEnd(log);
  return EXIT_SUCCESS;
}
