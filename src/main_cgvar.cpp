#include "types.h"

#include "apodizer.h"
#include "cg.h"
#include "cropper.h"
#include "fft_plan.h"
#include "filter.h"
#include "gridder.h"
#include "io_hd5.h"
#include "io_nifti.h"
#include "log.h"
#include "parse_args.h"
#include "sense.h"
#include "tensorOps.h"
#include "threads.h"

int main_cgvar(args::Subparser &parser)
{
  COMMON_RECON_ARGS;

  args::ValueFlag<float> thr(
      parser, "TRESHOLD", "Threshold for termination (1e-10)", {"thresh"}, 1.e-10);
  args::ValueFlag<long> its(
      parser, "MAX ITS", "Maximum number of iterations (8)", {'i', "max_its"}, 8);
  args::ValueFlag<float> iter_fov(
      parser, "ITER FOV", "Iterations FoV in mm (default 256 mm)", {"iter_fov"}, 256);
  args::ValueFlag<float> pre0(
      parser, "PRE0", "Preconditioning start value (default 1)", {"pre0"}, 1);
  args::ValueFlag<float> pre1(
      parser, "PRE1", "Preconditioning end value (default 1e-6)", {"pre1"}, 1.e-6f);

  Log log = ParseCommand(parser, iname);
  FFT::Start(log);

  HD5::Reader reader(iname.Get(), log);
  Trajectory const traj = reader.readTrajectory();
  auto const &info = traj.info();
  Cx3 rad_ks = info.noncartesianVolume();
  Kernel *kernel =
      kb ? (Kernel *)new KaiserBessel(kw.Get(), osamp.Get(), (info.type == Info::Type::ThreeD))
         : (Kernel *)new NearestNeighbour(kw ? kw.Get() : 1);
  Gridder gridder(traj.mapping(osamp.Get(), kernel->radius()), kernel, fastgrid, log);
  SDC::Load(sdc.Get(), traj, gridder, log);
  gridder.setSDCExponent(sdc_exp.Get());

  Cx4 grid = gridder.newMultichannel(info.channels);
  Cropper iter_cropper(info, gridder.gridDims(), iter_fov.Get(), log);
  Apodizer apodizer(kernel, gridder.gridDims(), iter_cropper.size(), log);
  FFT::ThreeDMulti fft(grid, log);

  long currentVolume = -1;
  Cx4 sense = iter_cropper.newMultichannel(info.channels);
  if (senseFile) {
    sense = LoadSENSE(senseFile.Get(), iter_cropper.dims(info.channels), log);
  } else {
    currentVolume = LastOrVal(senseVolume, info.volumes);
    reader.readNoncartesian(currentVolume, rad_ks);
    sense = DirectSENSE(traj, osamp.Get(), kernel, iter_fov.Get(), rad_ks, senseLambda.Get(), log);
  }

  Cx3 ones(1, info.read_points, info.spokes_total());
  ones.setConstant(1.0f);
  Cx4 transfer = gridder.newMultichannel(1);

  auto dev = Threads::GlobalDevice();
  CgVarSystem sys = [&](Cx3 const &x, Cx3 &y, float const pre) {
    auto const start = log.now();
    gridder.setSDCExponent(pre);
    transfer.device(dev) = transfer.constant(0.f);
    gridder.toCartesian(ones, transfer);

    grid.device(dev) = grid.constant(0.f);
    iter_cropper.crop4(grid).device(dev) = sense * Tile(x, info.channels);
    fft.forward(grid);
    grid.device(dev) = grid * transfer.broadcast(Sz4{info.channels, 1, 1, 1});
    fft.reverse(grid);

    y.device(dev) = (iter_cropper.crop4(grid) * sense.conjugate()).sum(Sz1{0});
    log.debug("System: {}", log.toNow(start));
  };

  DecodeFunction dec = [&](Cx3 const &x, Cx3 &y) {
    auto const &start = log.now();
    y.device(dev) = y.constant(0.f);
    gridder.setSDCExponent(1.f);
    gridder.toCartesian(x, grid);
    fft.reverse(grid);
    y.device(dev) = (iter_cropper.crop4(grid) * sense.conjugate()).sum(Sz1{0});
    apodizer.deapodize(y);
    log.debug("Decode: {}", log.toNow(start));
  };

  Cropper out_cropper(info, iter_cropper.size(), out_fov.Get(), log);
  Cx3 vol = iter_cropper.newImage();
  Cx3 cropped = out_cropper.newImage();
  Cx4 out = out_cropper.newSeries(info.volumes);
  auto const &all_start = log.now();
  for (long iv = 0; iv < info.volumes; iv++) {
    auto const &vol_start = log.now();
    if (iv != currentVolume) { // For single volume images, we already read it for SENSE
      reader.readNoncartesian(iv, rad_ks);
      currentVolume = iv;
    }
    dec(rad_ks, vol); // Initialize
    cgvar(sys, its.Get(), thr.Get(), pre0.Get(), pre1.Get(), vol, log);
    cropped = out_cropper.crop3(vol);
    if (tukey_s || tukey_e || tukey_h) {
      ImageTukey(tukey_s.Get(), tukey_e.Get(), tukey_h.Get(), cropped, log);
    }
    out.chip(iv, 3) = cropped;
    log.info("Volume {}: {}", iv, log.toNow(vol_start));
  }
  log.info("All Volumes: {}", log.toNow(all_start));
  WriteOutput(out, mag, false, info, iname.Get(), oname.Get(), "cgvar", oftype.Get(), log);
  FFT::End(log);
  return EXIT_SUCCESS;
}
