#include "types.h"

#include "apodizer.h"
#include "cg.h"
#include "cropper.h"
#include "fft3n.h"
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

  args::Flag magnitude(parser, "MAGNITUDE", "Output magnitude images only", {"magnitude"});
  args::ValueFlag<long> sense_vol(
      parser, "SENSE VOLUME", "Take SENSE maps from this volume", {"sense_vol"}, 0);
  args::ValueFlag<float> thr(
      parser, "TRESHOLD", "Threshold for termination (1e-10)", {"thresh"}, 1.e-10);
  args::ValueFlag<long> its(
      parser, "MAX ITS", "Maximum number of iterations (8)", {'i', "max_its"}, 8);
  args::ValueFlag<float> iter_fov(
      parser, "ITER FOV", "Iterations FoV in mm (default 256 mm)", {"iter_fov"}, 256);
  args::ValueFlag<float> pre0(
      parser, "PRE0", "Preconditioning start value (default 1)", {"pre0"}, 1);
  args::ValueFlag<float> pre1(parser, "PRE1", "Preconditioning end value (default 0)", {"pre1"}, 0);

  Log log = ParseCommand(parser, fname);
  FFT::Start(log);

  HD5::Reader reader(fname.Get(), log);
  Trajectory const traj = reader.readTrajectory();
  auto const &info = traj.info();
  Cx3 rad_ks = info.noncartesianVolume();
  Kernel *kernel =
      kb ? (Kernel *)new KaiserBessel(kw.Get(), osamp.Get(), (info.type == Info::Type::ThreeD))
         : (Kernel *)new NearestNeighbour(kw ? kw.Get() : 1);
  Gridder gridder(traj, osamp.Get(), kernel, fastgrid, log);
  SDC::Load(sdc.Get(), traj, gridder, log);
  gridder.setSDCExponent(sdc_exp.Get());

  Cx4 grid = gridder.newGrid();
  Cropper iter_cropper(info, gridder.gridDims(), iter_fov.Get(), log);
  FFT3N fft(grid, log);

  long currentVolume = SenseVolume(sense_vol, info.volumes);
  reader.readNoncartesian(currentVolume, rad_ks);
  Cx4 const sense = iter_cropper.crop4(SENSE(senseMethod.Get(), traj, gridder, rad_ks, log));

  CgVarSystem sys = [&](Cx3 const &x, Cx3 &y, float const pre) {
    auto const start = log.now();
    gridder.setSDCExponent(pre);
    grid.device(Threads::GlobalDevice()) = grid.constant(0.f);
    y = x;
    iter_cropper.crop4(grid).device(Threads::GlobalDevice()) = sense * Tile(y, info.channels);
    fft.forward();
    gridder.toNoncartesian(grid, rad_ks);
    grid.setZero();
    gridder.toCartesian(rad_ks, grid);
    fft.reverse();
    y.device(Threads::GlobalDevice()) = (iter_cropper.crop4(grid) * sense.conjugate()).sum(Sz1{0});
    log.debug("System: {}", log.toNow(start));
  };

  DecodeFunction dec = [&](Cx3 const &x, Cx3 &y) {
    auto const &start = log.now();
    y.setZero();
    gridder.setSDCExponent(1.f);
    grid.setZero();
    gridder.toCartesian(x, grid);
    fft.reverse();
    y.device(Threads::GlobalDevice()) = (iter_cropper.crop4(grid) * sense.conjugate()).sum(Sz1{0});
    log.debug("Decode: {}", log.toNow(start));
  };

  Cropper out_cropper(info, iter_cropper.size(), out_fov.Get(), log);
  Apodizer apodizer(kernel, gridder.gridDims(), out_cropper.size(), log);
  Cx3 vol = iter_cropper.newImage();
  Cx3 cropped = out_cropper.newImage();
  Cx4 out = out_cropper.newSeries(info.volumes);
  auto const &all_start = log.now();
  for (auto const &iv : WhichVolumes(volume.Get(), info.volumes)) {
    auto const &vol_start = log.now();
    if (iv != currentVolume) { // For single volume images, we already read it for SENSE
      reader.readNoncartesian(iv, rad_ks);
      currentVolume = iv;
    }
    dec(rad_ks, vol); // Initialize
    cgvar(sys, thr.Get(), its.Get(), pre0.Get(), pre1.Get(), vol, log);
    cropped = out_cropper.crop3(vol);
    apodizer.deapodize(cropped);
    if (tukey_s || tukey_e || tukey_h) {
      ImageTukey(tukey_s.Get(), tukey_e.Get(), tukey_h.Get(), cropped, log);
    }
    out.chip(iv, 3) = cropped;
    log.info("Volume {}: {}", iv, log.toNow(vol_start));
  }
  log.info("All Volumes: {}", log.toNow(all_start));
  auto const ofile = OutName(fname, oname, "cgvar");
  if (magnitude) {
    WriteVolumes(info, R4(out.abs()), volume.Get(), ofile, log);
  } else {
    WriteVolumes(info, out, volume.Get(), ofile, log);
  }
  FFT::End(log);
  return EXIT_SUCCESS;
}
