#include "types.h"

#include "cg.hpp"
#include "filter.h"
#include "log.h"
#include "op/recon-basis.h"
#include "op/recon.h"
#include "parse_args.h"
#include "sense.h"
#include "tensorOps.h"
#include "threads.h"

int main_cg(args::Subparser &parser)
{
  COMMON_RECON_ARGS;
  COMMON_SENSE_ARGS;

  args::ValueFlag<float> thr(
    parser, "TRESHOLD", "Threshold for termination (1e-10)", {"thresh"}, 1.e-10);
  args::ValueFlag<long> its(
    parser, "MAX ITS", "Maximum number of iterations (8)", {'i', "max_its"}, 8);
  args::ValueFlag<float> iter_fov(
    parser, "ITER FOV", "Iterations FoV in mm (default 256 mm)", {"iter_fov"}, 256);
  args::ValueFlag<std::string> basisFile(
    parser, "BASIS", "Read subspace basis from .h5 file", {"basis", 'b'});
  Log log = ParseCommand(parser, iname);
  FFT::Start(log);
  HD5::Reader reader(iname.Get(), log);
  Trajectory const traj = reader.readTrajectory();
  Info const &info = traj.info();
  auto gridder = make_grid(traj, osamp.Get(), kb, fastgrid, log);
  R2 const w = SDC::Choose(sdc.Get(), traj, gridder, log);
  gridder->setSDC(w);
  Cx4 senseMaps = senseFile ? LoadSENSE(senseFile.Get(), log)
                            : DirectSENSE(
                                info,
                                gridder.get(),
                                iter_fov.Get(),
                                senseLambda.Get(),
                                reader.noncartesian(ValOrLast(senseVol.Get(), info.volumes)),
                                log);

  if (basisFile) {
    HD5::Reader basisReader(basisFile.Get(), log);
    R2 const basis = basisReader.readBasis();
    long const nB = basis.dimension(1);
    auto basisGridder = make_grid_basis(gridder->mapping(), kb, fastgrid, basis, log);
    basisGridder->setSDC(w);
    ReconBasisOp recon(basisGridder.get(), senseMaps, log);
    auto sz = recon.dimensions();
    Cropper out_cropper(info, sz, out_fov.Get(), log);
    Cx4 vol(nB, sz[0], sz[1], sz[2]);
    Sz3 outSz = out_cropper.size();
    Cx4 cropped(nB, outSz[0], outSz[1], outSz[2]);
    Cx5 out(nB, outSz[0], outSz[1], outSz[2], info.volumes);
    auto const &all_start = log.now();
    for (long iv = 0; iv < info.volumes; iv++) {
      auto const &vol_start = log.now();
      recon.Adj(reader.noncartesian(iv), vol); // Initialize
      cg(its.Get(), thr.Get(), recon, vol, log);
      cropped = out_cropper.crop4(vol);
      out.chip(iv, 4) = cropped;
      log.info("Volume {}: {}", iv, log.toNow(vol_start));
    }
    log.info("All Volumes: {}", log.toNow(all_start));
    WriteBasisVolumes(out, basis, mag, info, iname.Get(), oname.Get(), "cg", oftype.Get(), log);
  } else {
    ReconOp recon(gridder.get(), senseMaps, log);
    recon.calcToeplitz(traj.info());
    Cx3 vol(recon.dimensions());
    Cropper out_cropper(info, vol.dimensions(), out_fov.Get(), log);
    Cx3 cropped = out_cropper.newImage();
    Cx4 out = out_cropper.newSeries(info.volumes);
    auto const &all_start = log.now();
    for (long iv = 0; iv < info.volumes; iv++) {
      auto const &vol_start = log.now();
      recon.Adj(reader.noncartesian(iv), vol); // Initialize
      cg(its.Get(), thr.Get(), recon, vol, log);
      cropped = out_cropper.crop3(vol);
      if (tukey_s || tukey_e || tukey_h) {
        ImageTukey(tukey_s.Get(), tukey_e.Get(), tukey_h.Get(), cropped, log);
      }
      out.chip(iv, 3) = cropped;
      log.info("Volume {}: {}", iv, log.toNow(vol_start));
    }
    log.info("All Volumes: {}", log.toNow(all_start));
    WriteOutput(out, mag, false, info, iname.Get(), oname.Get(), "cg", oftype.Get(), log);
  }
  FFT::End(log);
  return EXIT_SUCCESS;
}
