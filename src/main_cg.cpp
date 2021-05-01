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

int main_cg(args::Subparser &parser)
{
  COMMON_RECON_ARGS;

  args::ValueFlag<float> thr(
      parser, "TRESHOLD", "Threshold for termination (1e-10)", {"thresh"}, 1.e-10);
  args::ValueFlag<long> its(
      parser, "MAX ITS", "Maximum number of iterations (8)", {'i', "max_its"}, 8);
  args::ValueFlag<float> iter_fov(
      parser, "ITER FOV", "Iterations FoV in mm (default 256 mm)", {"iter_fov"}, 256);

  Log log = ParseCommand(parser, iname);
  FFT::Start(log);

  HD5::Reader reader(iname.Get(), log);
  Trajectory const traj = reader.readTrajectory();
  Info const &info = traj.info();
  Cx3 rad_ks = info.noncartesianVolume();

  Kernel *kernel =
      kb ? (Kernel *)new KaiserBessel(kw.Get(), osamp.Get(), (info.type == Info::Type::ThreeD))
         : (Kernel *)new NearestNeighbour(kw ? kw.Get() : 1);
  Gridder gridder(traj, osamp.Get(), kernel, fastgrid, log);
  SDC::Load(sdc.Get(), traj, gridder, log);
  gridder.setSDCExponent(sdc_exp.Get());

  Cx4 grid = gridder.newGrid();
  Cropper iter_cropper(info, gridder.gridDims(), iter_fov.Get(), log);
  FFT::ThreeDMulti fft(grid, log);

  long currentVolume;
  Cx4 const sense = LoadSENSE(
      info.channels,
      iter_cropper,
      senseFile.Get(),
      LastOrVal(senseVolume, info.volumes),
      reader,
      traj,
      osamp.Get(),
      kernel,
      senseLambda.Get(),
      rad_ks,
      currentVolume,
      log);

  Cx3 transfer(gridder.gridDims());
  {
    log.info("Calculating transfer function");
    Cx3 ones(1, info.read_points, info.spokes_total());
    ones.setConstant({1.0f});
    Cx4 transferTemp = gridder.newGridSingle();
    transfer.setZero();
    gridder.toCartesian(ones, transferTemp);
    transfer = transferTemp.reshape(gridder.gridDims());
  }

  CgSystem toe = [&](Cx3 const &x, Cx3 &y) {
    auto const start = log.now();
    grid.device(Threads::GlobalDevice()) = grid.constant(0.f);
    y = x;
    iter_cropper.crop4(grid).device(Threads::GlobalDevice()) = sense * Tile(y, info.channels);
    fft.forward(grid);
    grid.device(Threads::GlobalDevice()) = grid * Tile(transfer, info.channels);
    fft.reverse(grid);
    y.device(Threads::GlobalDevice()) = (iter_cropper.crop4(grid) * sense.conjugate()).sum(Sz1{0});
    log.debug("System: {}", log.toNow(start));
  };

  DecodeFunction dec = [&](Cx3 const &x, Cx3 &y) {
    auto const &start = log.now();
    y.setZero();
    grid.setZero();
    gridder.toCartesian(x, grid);
    fft.reverse(grid);
    y.device(Threads::GlobalDevice()) = (iter_cropper.crop4(grid) * sense.conjugate()).sum(Sz1{0});
    log.debug("Decode: {}", log.toNow(start));
  };

  Cropper out_cropper(info, iter_cropper.size(), out_fov.Get(), log);
  Apodizer apodizer(kernel, gridder.gridDims(), out_cropper.size(), log);
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
    cg(toe, its.Get(), thr.Get(), vol, log);
    cropped = out_cropper.crop3(vol);
    apodizer.deapodize(cropped);
    if (tukey_s || tukey_e || tukey_h) {
      ImageTukey(tukey_s.Get(), tukey_e.Get(), tukey_h.Get(), cropped, log);
    }
    out.chip(iv, 3) = cropped;
    log.info("Volume {}: {}", iv, log.toNow(vol_start));
  }
  log.info("All Volumes: {}", log.toNow(all_start));
  WriteOutput(out, mag, false, info, iname.Get(), oname.Get(), "cg", oftype.Get(), log);
  FFT::End(log);
  return EXIT_SUCCESS;
}
