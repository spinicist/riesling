#include "types.h"

#include "cropper.h"
#include "filter.h"
#include "log.h"
#include "op/grid.h"
#include "op/recon.hpp"
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
  args::ValueFlag<Index> its(
    parser, "MAX ITS", "Maximum number of iterations (16)", {'i', "max_its"}, 16);
  args::ValueFlag<float> iter_fov(
    parser, "ITER FOV", "Iterations FoV in mm (default 256 mm)", {"iter_fov"}, 256);
  args::ValueFlag<std::string> basisFile(
    parser, "BASIS", "Read subspace basis from .h5 file", {"basis", 'b'});
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

  if (basisFile) {
    HD5::Reader basisReader(basisFile.Get(), log);
    R2 const basis = basisReader.readBasis();
    gridder = make_grid_basis(gridder->mapping(), kernel.Get(), fastgrid, basis, log);
    gridder->setSDC(w);
  }
  gridder->setSDCPower(sdcPow.Get());
  ReconOp recon(gridder.get(), senseMaps, log);

  auto sz = recon.inputDimensions();
  Cropper out_cropper(info, Last3(sz), out_fov.Get(), log);
  Sz3 outSz = out_cropper.size();
  Cx5 out(sz[0], outSz[0], outSz[1], outSz[2], info.volumes);
  auto const &all_start = log.now();
  for (Index iv = 0; iv < info.volumes; iv++) {
    auto const &vol_start = log.now();
    out.chip<4>(iv) = out_cropper.crop4(tgv(
      its.Get(),
      thr.Get(),
      alpha.Get(),
      reduce.Get(),
      step_size.Get(),
      recon,
      reader.noncartesian(iv),
      log));
    log.info("Volume {}: {}", iv, log.toNow(vol_start));
  }
  log.info("All Volumes: {}", log.toNow(all_start));
  auto const fname = OutName(iname.Get(), oname.Get(), "tgv", "h5");
  HD5::Writer writer(fname, log);
  writer.writeInfo(info);
  writer.writeTensor(out, "image");
  FFT::End(log);

  return EXIT_SUCCESS;
}
