#include "types.h"

#include "cropper.h"
#include "filter.h"
#include "log.h"
#include "op/recon.h"
#include "parse_args.h"
#include "sense.h"
#include "tensorOps.h"
#include "tgv.h"

int main_tgv(args::Subparser &parser)
{
  COMMON_RECON_ARGS;
  COMMON_SENSE_ARGS;

  args::ValueFlag<float> thr(
      parser, "TRESHOLD", "Threshold for termination (1e-10)", {"thresh"}, 1.e-10);
  args::ValueFlag<long> its(
      parser, "MAX ITS", "Maximum number of iterations (16)", {'i', "max_its"}, 16);
  args::ValueFlag<float> iter_fov(
      parser, "ITER FOV", "Iterations FoV in mm (default 256 mm)", {"iter_fov"}, 256);
  args::ValueFlag<float> alpha(
      parser, "ALPHA", "Regularisation weighting (1e-5)", {"alpha"}, 1.e-5f);
  args::ValueFlag<float> reduce(
      parser, "REDUCE", "Reduce regularisation over iters (suggest 0.1)", {"reduce"}, 1.f);
  args::ValueFlag<float> step_size(
      parser, "STEP SIZE", "Inverse of step size (default 8)", {"step"}, 8.f);

  Log log = ParseCommand(parser, iname);
  FFT::Start(log);

  HD5::Reader reader(iname.Get(), log);
  Trajectory const traj = reader.readTrajectory();
  auto const &info = traj.info();
  Cx3 rad_ks = info.noncartesianVolume();

  auto gridder = make_grid(traj, osamp.Get(), kb, fastgrid, log);
  SDC::Choose(sdc.Get(), traj, gridder, log);
  gridder->setSDCExponent(sdc_exp.Get());

  Cx4 grid = gridder->newMultichannel(info.channels);
  grid.setZero();
  FFT::ThreeDMulti fft(grid, log);

  Cropper iter_cropper(info, gridder->gridDims(), iter_fov.Get(), log);
  R3 const apo = gridder->apodization(iter_cropper.size());
  long currentVolume = -1;
  Cx4 sense = iter_cropper.newMultichannel(info.channels);
  if (senseFile) {
    sense = LoadSENSE(senseFile.Get(), log);
  } else {
    currentVolume = LastOrVal(senseVolume, info.volumes);
    reader.readNoncartesian(currentVolume, rad_ks);
    sense = DirectSENSE(traj, osamp.Get(), kb, iter_fov.Get(), rad_ks, senseLambda.Get(), log);
  }

  ReconOp recon(traj, osamp.Get(), kb, fastgrid, sdc.Get(), sense, log);
  recon.setPreconditioning(sdc_exp.Get());

  Cropper out_cropper(info, iter_cropper.size(), out_fov.Get(), log);
  Cx3 image = out_cropper.newImage();
  Cx4 out = out_cropper.newSeries(info.volumes);
  for (long iv = 0; iv < info.volumes; iv++) {
    auto const start = log.now();
    log.info(FMT_STRING("Processing volume: {}"), iv);
    if (iv != currentVolume) { // For single volume images, we already read it for SENSE
      reader.readNoncartesian(iv, rad_ks);
      currentVolume = iv;
    }
    image = out_cropper.crop3(
        tgv(its.Get(), thr.Get(), alpha.Get(), reduce.Get(), step_size.Get(), recon, rad_ks, log));
    if (tukey_s || tukey_e || tukey_h) {
      ImageTukey(tukey_s.Get(), tukey_e.Get(), tukey_h.Get(), image, log);
    }
    out.chip(iv, 3) = image;
    log.info("Volume {}: {}", iv, log.toNow(start));
  }
  WriteOutput(out, mag, false, info, iname.Get(), oname.Get(), "tgv", oftype.Get(), log);
  FFT::End(log);
  return EXIT_SUCCESS;
}
