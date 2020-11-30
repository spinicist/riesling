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
  args::Positional<std::string> fname(parser, "FILE", "HD5 file to recon");

  args::ValueFlag<std::string> oname(parser, "OUTPUT", "Override output name", {'o', "out"});
  args::ValueFlag<long> volume(parser, "VOLUME", "Only recon this volume", {"vol"}, -1);
  args::ValueFlag<float> crop(
      parser, "CROP SIZE", "Crop extent in mm (default header FoV)", {"crop"}, -1);
  args::Flag magnitude(parser, "MAGNITUDE", "Output magnitude images only", {"magnitude"});
  args::Flag est_dc(parser, "ESTIMATE DC", "Estimate DC weights instead of analytic", {"est_dc"});
  args::ValueFlag<float> dc_exp(
      parser, "DC Exponent", "Density-Compensation Exponent (default 1.0)", {'d', "dce"}, 1.0f);
  args::Flag save_maps(parser, "SAVE MAPS", "Write out sensitivity maps", {"maps", 'm'});
  args::Flag save_channels(
      parser, "SAVE CHANNELS", "Write out individual channel images", {"channels", 'c'});
  args::ValueFlag<float> osamp(
      parser, "GRID OVERSAMPLE", "Oversampling factor for gridding, default 2", {'g', "grid"}, 2.f);
  args::ValueFlag<long> sense_vol(
      parser, "SENSE VOLUME", "Take SENSE maps from this volume (default last)", {"sense_vol"}, -1);
  args::ValueFlag<float> tukey_s(
      parser, "TUKEY START", "Start-width of Tukey filter", {"tukey_start"}, 1.0f);
  args::ValueFlag<float> tukey_e(
      parser, "TUKEY END", "End-width of Tukey filter", {"tukey_end"}, 1.0f);
  args::ValueFlag<float> tukey_h(
      parser, "TUKEY HEIGHT", "End height of Tukey filter", {"tukey_height"}, 0.0f);

  Log log = ParseCommand(parser, fname);
  FFTStart(log);

  RadialReader reader(fname.Get(), log);
  auto const &info = reader.info();
  auto const trajectory = reader.readTrajectory();

  Gridder gridder(info, trajectory, osamp.Get(), log);
  gridder.setDCExponent(dc_exp.Get());
  if (est_dc) {
    gridder.estimateDC();
  }
  Cx4 grid = gridder.newGrid();
  grid.setZero();
  FFT3N fft(grid, log);

  Cropper cropper(info, gridder.gridDims(), crop.Get(), log);
  Cx3 rad_ks = info.radialVolume();
  reader.readData(SenseVolume(sense_vol, info.volumes), rad_ks);
  Cx4 sense = cropper.crop4(SENSE(info, trajectory, osamp.Get(), rad_ks, log));
  if (save_maps) {
    WriteNifti(info, Cx4(sense.shuffle(Sz4{1, 2, 3, 0})), OutName(fname, oname, "sense-maps"), log);
  }

  Cx3 image = cropper.newImage();
  Cx4 out = cropper.newSeries(info.volumes);
  Cx4 channel_images = cropper.newMultichannel(info.channels);
  auto const &all_start = log.start_time();
  bool channels_saved = false;
  for (auto const &iv : WhichVolumes(volume.Get(), info.volumes)) {
    log.info(FMT_STRING("Processing Echo: {}"), iv);
    auto const &vol_start = log.start_time();
    if (info.volumes > 1) { // Might be different to SENSE volume
      reader.readData(iv, rad_ks);
    }
    grid.setZero();
    gridder.toCartesian(rad_ks, grid);
    fft.reverse();
    channel_images.device(Threads::GlobalDevice()) = cropper.crop4(grid) * sense.conjugate();
    image.device(Threads::GlobalDevice()) = channel_images.sum(Sz1{0});
    fmt::print("chans {} image {}\n", channel_images.dimensions(), image.dimensions());
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
