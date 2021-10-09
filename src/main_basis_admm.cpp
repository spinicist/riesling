#include "types.h"

#include "admm.hpp"
#include "cropper.h"
#include "fft_plan.h"
#include "filter.h"
#include "llr.h"
#include "log.h"
#include "op/grid.h"
#include "op/recon-basis.h"
#include "op/sense.h"
#include "parse_args.h"
#include "sense.h"

int main_basis_admm(args::Subparser &parser)
{
  COMMON_RECON_ARGS;
  COMMON_SENSE_ARGS;
  args::ValueFlag<float> thr(parser, "T", "Threshold for termination (1e-10)", {"thresh"}, 1.e-10);
  args::ValueFlag<long> lsq_its(parser, "ITS", "Inner iterations (8)", {"lsq_its"}, 8);
  args::ValueFlag<long> admm_its(parser, "ITS", "Outer iterations (8)", {"admm_its"}, 8);
  args::ValueFlag<float> iter_fov(
      parser, "FOV", "Iterations FoV in mm (default 256 mm)", {"iter_fov"}, 256);
  args::ValueFlag<float> reg_lambda(parser, "L", "ADMM lambda (default 0.1)", {"reg"}, 0.1f);
  args::ValueFlag<float> reg_rho(parser, "R", "ADMM rho (default 0.1)", {"rho"}, 0.1f);
  args::ValueFlag<long> patch(parser, "P", "Patch size for LLR (default 8)", {"patch"}, 8);
  args::ValueFlag<std::string> basisFile(
      parser, "BASIS", "Read subspace basis from .h5 file", {"basis", 'b'});

  Log log = ParseCommand(parser, iname);
  FFT::Start(log);

  HD5::Reader reader(iname.Get(), log);
  Trajectory const traj = reader.readTrajectory();
  Info const &info = traj.info();
  Cx3 rad_ks = info.noncartesianVolume();

  long nB = 1;
  R2 basis(1, 1);
  basis.setConstant(1.f);
  if (basisFile) {
    HD5::Reader basisReader(basisFile.Get(), log);
    basis = basisReader.readBasis();
    nB = basis.dimension(1);

    if ((info.spokes_total() % basis.dimension(0)) != 0) {
      Log::Fail(
          FMT_STRING("Basis length {} does not evenly divide number of spokes {}"),
          basis.dimension(0),
          info.spokes_total());
    }
    log.info("Basis has {} vectors with {} entries", nB, basis.dimension(0));
  }

  long currentVolume = -1;
  Cx4 senseMaps;
  if (senseFile) {
    senseMaps = LoadSENSE(senseFile.Get(), log);
  } else {
    currentVolume = LastOrVal(senseVolume, info.volumes);
    reader.readNoncartesian(currentVolume, rad_ks);
    senseMaps = DirectSENSE(traj, osamp.Get(), kb, iter_fov.Get(), rad_ks, senseLambda.Get(), log);
  }

  ReconBasisOp recon(traj, osamp.Get(), kb, fastgrid, sdc.Get(), senseMaps, basis, log);
  recon.setPreconditioning(sdc_exp.Get());

  auto reg = [&](Cx4 const &x) -> Cx4 { return llr(x, reg_lambda.Get(), patch.Get(), log); };

  auto sz = recon.dimensions();
  Cropper out_cropper(info, sz, out_fov.Get(), log);
  Cx4 vol(nB, sz[0], sz[1], sz[2]);
  Sz3 outSz = out_cropper.size();
  Cx4 cropped(nB, outSz[0], outSz[1], outSz[2]);
  Cx5 out(nB, outSz[0], outSz[1], outSz[2], info.volumes);
  auto const &all_start = log.now();
  for (long iv = 0; iv < info.volumes; iv++) {
    auto const &vol_start = log.now();
    if (iv != currentVolume) { // For single volume images, we already read it for senseMaps
      reader.readNoncartesian(iv, rad_ks);
      currentVolume = iv;
    }
    recon.Adj(rad_ks, vol); // Initialize
    admm(admm_its.Get(), lsq_its.Get(), thr.Get(), recon, reg_rho.Get(), reg, vol, log);
    cropped = out_cropper.crop4(vol);
    out.chip(iv, 4) = cropped;
    log.info("Volume {}: {}", iv, log.toNow(vol_start));
  }
  log.info("All Volumes: {}", log.toNow(all_start));
  WriteBasisVolumes(
      out, basis, mag, false, info, iname.Get(), oname.Get(), "admm", oftype.Get(), log);
  FFT::End(log);
  return EXIT_SUCCESS;
}
