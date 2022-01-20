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
  args::ValueFlag<Index> its(
    parser, "MAX ITS", "Maximum number of iterations (8)", {'i', "max_its"}, 8);
  args::ValueFlag<float> iter_fov(
    parser, "ITER FOV", "Iterations FoV in mm (default 256 mm)", {"iter_fov"}, 256);
  args::ValueFlag<float> pre0(
    parser, "PRE0", "Preconditioning start value (default 1)", {"pre0"}, 1);
  args::ValueFlag<float> pre1(
    parser, "PRE1", "Preconditioning end value (default 1e-6)", {"pre1"}, 1.e-6f);

  ParseCommand(parser, iname);
  FFT::Start();

  HD5::Reader reader(iname.Get());
  Trajectory const traj = reader.readTrajectory();
  Info const &info = traj.info();

  Cx4 senseMaps;
  if (senseFile) {
    senseMaps = LoadSENSE(senseFile.Get());
  } else {
    senseMaps =
      DirectSENSE(traj, osamp.Get(), kb, iter_fov.Get(), senseLambda.Get(), senseVol.Get(), reader);
  }

  ReconOp recon(traj, osamp.Get(), kb, fastgrid, sdc.Get(), senseMaps);
  recon.setPreconditioning(pre0);
  recon.calcToeplitz(traj.info());
  Cx3 vol(recon.dimensions());
  Cropper out_cropper(info, vol.dimensions(), out_fov.Get());
  Cx3 cropped = out_cropper.newImage();
  Cx4 out = out_cropper.newSeries(info.volumes);
  auto const &all_start = Log::Now();
  for (Index iv = 0; iv < info.volumes; iv++) {
    auto const &vol_start = Log::Now();
    recon.Adj(reader.noncartesian(iv), vol); // Initialize
    cgvar(its.Get(), thr.Get(), pre0.Get(), pre1.Get(), recon, vol);
    cropped = out_cropper.crop3(vol);
    if (tukey_s || tukey_e || tukey_h) {
      ImageTukey(tukey_s.Get(), tukey_e.Get(), tukey_h.Get(), cropped);
    }
    out.chip<3>(iv) = cropped;
    Log::Print("Volume {}: {}", iv, Log::ToNow(vol_start));
  }
  Log::Print("All Volumes: {}", Log::ToNow(all_start));
  WriteOutput(out, mag, false, info, iname.Get(), oname.Get(), "cgvar", "h5");
  FFT::End();
  return EXIT_SUCCESS;
}
