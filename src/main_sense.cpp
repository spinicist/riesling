#include "types.h"

#include "apodizer.h"
#include "cropper.h"
#include "espirit.h"
#include "fft3n.h"
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
  args::ValueFlag<float> thresh(
      parser, "THRESHOLD", "Threshold for removing background", {"thresh"}, 0.f);
  args::Flag save_maps(parser, "MAPS", "Write out sensitivity maps", {"maps", 'm'});
  args::Flag save_kernels(parser, "KERNELS", "Write out k-space kernels", {"kernels", 'k'});
  args::Flag save_channels(
      parser, "CHANNELS", "Write out individual channel images", {"channels", 'c'});
  args::Flag espirit(parser, "ESPIRIT", "Use ESPIRIT", {"espirit"});
  args::ValueFlag<long> kernelSz(
      parser, "KERNEL SIZE", "ESPIRIT Kernel size (default 6)", {"kernel"}, 6);
  args::ValueFlag<long> calSz(
      parser, "CAL SIZE", "ESPIRIT Calibration region size (default 32)", {"cal"}, 32);
  args::ValueFlag<float> retain(
      parser,
      "RETAIN",
      "ESPIRIT Fraction of singular vectors to retain (default 0.25)",
      {"retain"},
      0.25);

  Log log = ParseCommand(parser, fname);
  FFT::Start(log);

  HD5::Reader reader(fname.Get(), log);
  auto const traj = reader.readTrajectory();
  auto const &info = traj.info();
  Kernel *kernel =
      kb ? (Kernel *)new KaiserBessel(3, osamp.Get(), (info.type == Info::Type::ThreeD))
         : (Kernel *)new NearestNeighbour();
  Gridder gridder(traj, osamp.Get(), kernel, fastgrid, log);
  SDC::Load(sdc.Get(), traj, gridder, log);
  gridder.setSDCExponent(sdc_exp.Get());

  Cropper cropper(info, gridder.gridDims(), out_fov.Get(), log);
  Apodizer apodizer(kernel, gridder.gridDims(), cropper.size(), log);
  Cx3 rad_ks = info.noncartesianVolume();
  long currentVolume = SenseVolume(sense_vol, info.volumes);
  reader.readNoncartesian(currentVolume, rad_ks);
  Cx4 sense = cropper.crop4(SENSE(senseMethod.Get(), traj, gridder, rad_ks, log));
  if (save_maps) {
    WriteNifti(
        info,
        Cx4(sense.shuffle(Sz4{1, 2, 3, 0})),
        OutName(fname, oname, "sense-maps", outftype.Get()),
        log);
  }
  if (save_kernels) {
    FFT3N kernelFFT(sense, log);
    kernelFFT.forward();
    WriteNifti(
        info,
        Cx4(sense.shuffle(Sz4{1, 2, 3, 0})),
        OutName(fname, oname, "sense-kernels", outftype.Get()),
        log);
    kernelFFT.reverse();
  }

  Cx4 grid = gridder.newGrid();
  grid.setZero();
  FFT3N fft(grid, log);
  Cx3 image = cropper.newImage();
  Cx4 out = cropper.newSeries(info.volumes);
  Cx4 channel_images = cropper.newMultichannel(info.channels);
  auto const &all_start = log.now();
  bool channels_saved = false;
  for (auto const &iv : WhichVolumes(volume.Get(), info.volumes)) {
    log.info(FMT_STRING("Processing volume: {}"), iv);
    auto const &vol_start = log.now();
    if (iv != currentVolume) { // For single volume images, we already read it for SENSE
      reader.readNoncartesian(iv, rad_ks);
      currentVolume = iv;
    }
    grid.setZero();
    gridder.toCartesian(rad_ks, grid);
    fft.reverse();
    channel_images.device(Threads::GlobalDevice()) = cropper.crop4(grid) * sense.conjugate();
    image.device(Threads::GlobalDevice()) = channel_images.sum(Sz1{0});
    apodizer.deapodize(image);
    if (tukey_s || tukey_e || tukey_h) {
      ImageTukey(tukey_s.Get(), tukey_e.Get(), tukey_h.Get(), image, log);
    }

    out.chip(iv, 3) = image;
    log.info("Volume {}: {}", iv, log.toNow(vol_start));
    if (save_channels && !channels_saved) {
      WriteNifti(
          info,
          Cx4(channel_images.shuffle(Sz4{1, 2, 3, 0})),
          OutName(fname, oname, "sense-channels"),
          log);
      channels_saved = true;
    }
  }
  log.info("All Volumes: {}", log.toNow(all_start));

  auto const ofile = OutName(fname, oname, "sense", outftype.Get());
  if (magnitude) {
    WriteVolumes(info, R4(out.abs()), volume.Get(), ofile, log);
  } else {
    WriteVolumes(info, out, volume.Get(), ofile, log);
  }

  FFT::End(log);
  return EXIT_SUCCESS;
}
