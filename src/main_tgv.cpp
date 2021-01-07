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
  COMMON_RECON_ARGS;

  args::Flag magnitude(parser, "MAGNITUDE", "Output magnitude images only", {"magnitude"});
  args::ValueFlag<long> sense_vol(
      parser, "SENSE VOLUME", "Take SENSE maps from this volume", {"sense_vol"}, 0);
  args::ValueFlag<float> thr(
      parser, "TRESHOLD", "Threshold for termination (1e-10)", {"thresh"}, 1.e-10);
  args::ValueFlag<long> its(
      parser, "MAX ITS", "Maximum number of iterations (16)", {'i', "max_its"}, 16);
  args::ValueFlag<float> iter_fov(
      parser, "ITER FOV", "Iterations FoV in mm (default 256 mm)", {"iter_fov"}, 256);
  args::ValueFlag<float> l1_weight(
      parser, "L1", "L1-Regularisation weighting (1e-5)", {"l1"}, 1.e-5f);
  args::ValueFlag<float> l1_reduction(
      parser, "L1 REDUCTION", "Reduce L1 by factor over iters (suggest 0.1)", {"l1reduce"}, 1.f);
  args::ValueFlag<float> step_size(
      parser, "STEP SIZE", "Inverse of step size (default 8)", {"step"}, 8.f);

  Log log = ParseCommand(parser, fname);
  FFTStart(log);

  HD5Reader reader(fname.Get(), log);
  auto const info = reader.info();
  auto const trajectory = reader.readTrajectory();

  Gridder gridder(info, trajectory, osamp.Get(), est_dc, kb, stack, log);
  gridder.setDCExponent(dc_exp.Get());
  Cx4 grid = gridder.newGrid();
  grid.setZero();
  FFT3N fft(grid, log);

  Cropper iter_cropper(info, gridder.gridDims(), iter_fov.Get(), stack, log);
  Cx3 rad_ks = info.noncartesianVolume();
  long currentVolume = SenseVolume(sense_vol, info.volumes);
  reader.readData(currentVolume, rad_ks);
  Cx4 sense = iter_cropper.crop4(SENSE(info, trajectory, osamp.Get(), stack, kb, rad_ks, log));

  EncodeFunction enc = [&](Cx3 &x, Cx3 &y) {
    auto const &start = log.start_time();
    y.setZero();
    grid.setZero();
    gridder.apodize(x);
    iter_cropper.crop4(grid).device(Threads::GlobalDevice()) = tile(x, info.channels) * sense;
    fft.forward();
    gridder.toNoncartesian(grid, y);
    log.stop_time(start, "Total encode time");
  };

  DecodeFunction dec = [&](Cx3 const &x, Cx3 &y) {
    auto const &start = log.start_time();
    grid.setZero();
    gridder.toCartesian(x, grid);
    fft.reverse();
    y.device(Threads::GlobalDevice()) = (iter_cropper.crop4(grid) * sense.conjugate()).sum(Sz1{0});
    gridder.deapodize(y);
    log.stop_time(start, "Total decode time");
  };

  Cropper out_cropper(info, iter_cropper.size(), out_fov.Get(), stack, log);
  Cx4 out = out_cropper.newSeries(info.volumes);
  for (auto const &iv : WhichVolumes(volume.Get(), info.volumes)) {
    auto const start = log.start_time();
    log.info(FMT_STRING("Processing volume: {}"), iv);
    if (iv != currentVolume) { // For single volume images, we already read it for SENSE
      reader.readData(iv, rad_ks);
      currentVolume = iv;
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
