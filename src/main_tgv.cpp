#include "types.h"

#include "apodizer.h"
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
  args::ValueFlag<float> alpha(
      parser, "ALPHA", "Regularisation weighting (1e-5)", {"alpha"}, 1.e-5f);
  args::ValueFlag<float> reduce(
      parser, "REDUCE", "Reduce regularisation over iters (suggest 0.1)", {"reduce"}, 1.f);
  args::ValueFlag<float> step_size(
      parser, "STEP SIZE", "Inverse of step size (default 8)", {"step"}, 8.f);

  Log log = ParseCommand(parser, fname);
  FFT::Start(log);

  HD5::Reader reader(fname.Get(), log);
  Trajectory const traj = reader.readTrajectory();
  auto const &info = traj.info();

  Kernel *kernel =
      kb ? (Kernel *)new KaiserBessel(kw.Get(), osamp.Get(), (info.type == Info::Type::ThreeD))
         : (Kernel *)new NearestNeighbour(kw ? kw.Get() : 1);
  Gridder gridder(traj, osamp.Get(), kernel, fastgrid, log);
  SDC::Load(sdc.Get(), traj, gridder, log);
  gridder.setSDCExponent(sdc_exp.Get());

  Cx4 grid = gridder.newGrid();
  grid.setZero();
  FFT3N fft(grid, log);

  Cropper iter_cropper(info, gridder.gridDims(), iter_fov.Get(), log);
  Cx3 rad_ks = info.noncartesianVolume();
  long currentVolume = SenseVolume(sense_vol, info.volumes);
  reader.readNoncartesian(currentVolume, rad_ks);
  Cx4 sense = iter_cropper.crop4(SENSE(senseMethod.Get(), traj, gridder, rad_ks, log));

  EncodeFunction enc = [&](Cx3 &x, Cx3 &y) {
    auto const &start = log.now();
    y.setZero();
    grid.setZero();
    iter_cropper.crop4(grid).device(Threads::GlobalDevice()) = Tile(x, info.channels) * sense;
    fft.forward();
    gridder.toNoncartesian(grid, y);
    log.debug("Encode: {}", log.toNow(start));
  };

  DecodeFunction dec = [&](Cx3 const &x, Cx3 &y) {
    auto const &start = log.now();
    grid.setZero();
    gridder.toCartesian(x, grid);
    fft.reverse();
    y.device(Threads::GlobalDevice()) = (iter_cropper.crop4(grid) * sense.conjugate()).sum(Sz1{0});
    log.debug("Decode: {}", log.toNow(start));
  };

  Cropper out_cropper(info, iter_cropper.size(), out_fov.Get(), log);
  Apodizer apodizer(kernel, gridder.gridDims(), out_cropper.size(), log);
  Cx3 image = out_cropper.newImage();
  Cx4 out = out_cropper.newSeries(info.volumes);
  for (auto const &iv : WhichVolumes(volume.Get(), info.volumes)) {
    auto const start = log.now();
    log.info(FMT_STRING("Processing volume: {}"), iv);
    if (iv != currentVolume) { // For single volume images, we already read it for SENSE
      reader.readNoncartesian(iv, rad_ks);
      currentVolume = iv;
    }
    image = out_cropper.crop3(
        tgv(rad_ks,
            iter_cropper.size(),
            enc,
            dec,
            its.Get(),
            thr.Get(),
            alpha.Get(),
            reduce.Get(),
            step_size.Get(),
            log));
    apodizer.deapodize(image);
    if (tukey_s || tukey_e || tukey_h) {
      ImageTukey(tukey_s.Get(), tukey_e.Get(), tukey_h.Get(), image, log);
    }

    out.chip(iv, 3) = image;
    log.info("Volume {}: {}", iv, log.toNow(start));
  }
  auto const ofile = OutName(fname, oname, "tgv", outftype.Get());
  if (magnitude) {
    WriteVolumes(info, R4(out.abs()), volume.Get(), ofile, log);
  } else {
    WriteVolumes(info, out, volume.Get(), ofile, log);
  }
  FFT::End(log);
  return EXIT_SUCCESS;
}
