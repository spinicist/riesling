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
#include "tgv.h"

int main_tgv(args::Subparser &parser)
{
  args::Positional<std::string> fname(parser, "FILE", "HD5 file to recon");
  args::ValueFlag<std::string> oname(parser, "OUTPUT", "Override output name", {'o', "out"});

  args::ValueFlag<long> volume(parser, "VOLUME", "Only recon this volume", {"vol"}, -1);
  args::ValueFlag<float> crop(
      parser, "CROP SIZE", "Crop extent in mm (default 220)", {"crop"}, 220.f);
  args::Flag est_dc(parser, "ESTIMATE DC", "Estimate DC weights instead of analytic", {"est_dc"});
  args::ValueFlag<float> dc_exp(
      parser, "DC Exponent", "Density-Compensation Exponent (default 1.0)", {'d', "dce"}, 1.0f);
  args::ValueFlag<float> osamp(
      parser, "GRID OVERSAMPLE", "Oversampling factor for gridding, default 2", {'g', "grid"}, 2.f);
  args::ValueFlag<long> sense_vol(
      parser, "SENSE VOLUME", "Take SENSE maps from this volume", {"sense_vol"}, 0);

  args::ValueFlag<float> thr(
      parser, "TRESHOLD", "Threshold for termination (1e-10)", {"thresh"}, 1.e-10);
  args::ValueFlag<long> its(
      parser, "MAX ITS", "Maximum number of iterations (16)", {'i', "max_its"}, 16);
  args::ValueFlag<float> l1_weight(
      parser, "L1", "L1-Regularisation weighting (1e-5)", {"l1"}, 1.e-5f);
  args::ValueFlag<float> l1_reduction(
      parser, "L1 REDUCTION", "Reduce L1 by factor over iters (suggest 0.1)", {"l1reduce"}, 1.f);
  args::ValueFlag<float> step_size(
      parser, "STEP SIZE", "Inverse of step size (default 8)", {"step"}, 8.f);

  args::ValueFlag<float> tukey_s(
      parser, "TUKEY START", "Start-width of Tukey filter", {"tukey_start"}, 1.0f);
  args::ValueFlag<float> tukey_e(
      parser, "TUKEY END", "End-width of Tukey filter", {"tukey_end"}, 1.0f);
  args::ValueFlag<float> tukey_h(
      parser, "TUKEY HEIGHT", "End height of Tukey filter", {"tukey_height"}, 0.0f);

  args::Flag magnitude(parser, "MAGNITUDE", "Output magnitude images only", {"magnitude"});
  Log log = ParseCommand(parser, fname);
  FFTStart(log);

  RadialReader reader(fname.Get(), log);
  auto const info = reader.info();
  auto const trajectory = reader.readTrajectory();

  Gridder gridder(info, trajectory, osamp.Get(), log);
  gridder.setDCExponent(dc_exp.Get());
  if (est_dc) {
    gridder.estimateDC();
  }
  Cx4 grid = gridder.newGrid();
  grid.setZero();
  FFT3N fft(grid, log);

  Cropper iter_cropper(info, gridder.gridDims(), crop.Get(), log);
  Cx3 rad_ks = info.radialVolume();
  reader.readData(SenseVolume(sense_vol, info.volumes), rad_ks);
  Cx4 sense = iter_cropper.crop4(SENSE(info, trajectory, osamp.Get(), rad_ks, log));

  EncodeFunction enc = [&](Cx3 const &x, Cx3 &radial) {
    auto const &start = log.start_time();
    radial.setZero();
    grid.setZero();
    iter_cropper.crop4(grid).device(Threads::GlobalDevice()) = tile(x, info.channels) * sense;
    fft.forward();
    gridder.toRadial(grid, radial);
    log.stop_time(start, "Total encode time");
  };

  DecodeFunction dec = [&](Cx3 const &radial, Cx3 &y) {
    auto const &start = log.start_time();
    grid.setZero();
    gridder.toCartesian(radial, grid);
    fft.reverse();
    y.device(Threads::GlobalDevice()) = (iter_cropper.crop4(grid) * sense.conjugate()).sum(Sz1{0});
    log.stop_time(start, "Total decode time");
  };

  Cropper out_cropper(info, iter_cropper.size(), -1, log);
  Cx4 out = out_cropper.newSeries(info.volumes);
  for (auto const &iv : WhichVolumes(volume.Get(), info.volumes)) {
    auto const start = log.start_time();
    log.info(FMT_STRING("Processing Echo: {}"), iv);
    if (info.volumes > 1) { // Might be different to SENSE volume
      reader.readData(iv, rad_ks);
    }
    Cx3 image = out_cropper.crop3(
        tgv(rad_ks,
            iter_cropper.size(),
            enc,
            dec,
            its.Get(),
            thr.Get(),
            l1_weight.Get(),
            l1_reduction.Get(),
            step_size.Get(),
            log));

    if (tukey_s || tukey_e || tukey_h) {
      ImageTukey(tukey_s.Get(), tukey_e.Get(), tukey_h.Get(), image, log);
    }

    out.chip(iv, 3) = image;
    log.stop_time(start, "Total volume time");
  }
  auto const ofile = OutName(fname, oname, "tgv");
  if (magnitude) {
    WriteVolumes(info, R4(out.abs()), volume.Get(), ofile, log);
  } else {
    WriteVolumes(info, out, volume.Get(), ofile, log);
  }
  FFTEnd(log);
  return EXIT_SUCCESS;
}
