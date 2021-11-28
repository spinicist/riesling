#include "types.h"

#include "cg.hpp"
#include "filter.h"
#include "log.h"
#include "op/grid-basis.h"
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
  auto gridder = make_grid(traj, osamp.Get(), kernel.Get(), fastgrid, log);
  R2 const w = SDC::Choose(sdc.Get(), traj, osamp.Get(), log);
  gridder->setSDC(w);
  Cx4 senseMaps = senseFile ? LoadSENSE(senseFile.Get(), log)
                            : DirectSENSE(
                                info,
                                gridder.get(),
                                iter_fov.Get(),
                                senseLambda.Get(),
                                reader.noncartesian(ValOrLast(senseVol.Get(), info.volumes)),
                                log);

  std::unique_ptr<GridBase> gridder2;
  long n1; // Size of first dimension
  if (basisFile) {
    HD5::Reader basisReader(basisFile.Get(), log);
    R2 const basis = basisReader.readBasis();
    n1 = basis.dimension(1);
    gridder2 = make_grid_basis(gridder->mapping(), kernel.Get(), fastgrid, basis, log);
    gridder2->setSDC(w);
  } else {
    gridder2 = std::move(gridder);
    n1 = info.echoes;
  }

  ReconBasisOp recon(gridder2.get(), senseMaps, log);
  if (!basisFile) {
    recon.calcToeplitz(traj.info());
  }
  auto sz = recon.dimensions();
  Cropper out_cropper(info, sz, out_fov.Get(), log);
  Cx4 vol(n1, sz[0], sz[1], sz[2]);
  Sz3 outSz = out_cropper.size();
  Cx4 cropped(n1, outSz[0], outSz[1], outSz[2]);
  Cx5 out(n1, outSz[0], outSz[1], outSz[2], info.volumes);
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

  if (basisFile) {
    WriteBasisVolumes(out, basis, mag, info, iname.Get(), oname.Get(), "cg", oftype.Get(), log);
  } else {
    WriteOutput(out, mag, false, info, iname.Get(), oname.Get(), "cg", oftype.Get(), log);
  }

  FFT::End(log);
  return EXIT_SUCCESS;
}
