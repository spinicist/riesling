#include "types.h"

#include "cg.hpp"
#include "cropper.h"
#include "filter.h"
#include "log.h"
#include "op/recon.h"
#include "parse_args.h"
#include "sense.h"
#include "tensorOps.h"
#include "threads.h"

int main_cgvar(args::Subparser &parser)
{
  COMMON_RECON_ARGS;
  COMMON_SENSE_ARGS;

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
  Info const &info = traj.info();

  Cx4 senseMaps;
  if (senseFile) {
    senseMaps = LoadSENSE(senseFile.Get(), log);
  } else {
    senseMaps = DirectSENSE(
      traj, osamp.Get(), kb, iter_fov.Get(), senseLambda.Get(), senseVol.Get(), reader, log);
  }

  ReconOp recon(traj, osamp.Get(), kb, fastgrid, sdc.Get(), senseMaps, log);
  recon.setPreconditioning(sdc_exp.Get());
  recon.calcToeplitz(traj.info());
  Cx3 vol(recon.dimensions());
  Cropper out_cropper(info, vol.dimensions(), out_fov.Get(), log);
  Cx3 cropped = out_cropper.newImage();
  Cx4 out = out_cropper.newSeries(info.volumes);
  auto const &all_start = log.now();
  for (long iv = 0; iv < info.volumes; iv++) {
    auto const &vol_start = log.now();
    recon.Adj(reader.noncartesian(iv), vol); // Initialize
    cgvar(its.Get(), thr.Get(), pre0.Get(), pre1.Get(), recon, vol, log);
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
