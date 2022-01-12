#include "types.h"

#include "cropper.h"
#include "fft_plan.h"
#include "filter.h"
#include "io.h"
#include "log.h"
#include "op/grid.h"
#include "parse_args.h"
#include "sense.h"
#include "tensorOps.h"

int main_recon(args::Subparser &parser)
{
  COMMON_RECON_ARGS;
  COMMON_SENSE_ARGS;
  args::Flag rss(parser, "RSS", "Use Root-Sum-Squares channel combination", {"rss", 'r'});
  args::ValueFlag<std::string> basisFile(
    parser, "BASIS", "Read subspace basis from .h5 file", {"basis", 'b'});

  Log log = ParseCommand(parser, iname);
  FFT::Start(log);
  HD5::Reader reader(iname.Get(), log);
  auto const traj = reader.readTrajectory();
  auto const &info = traj.info();
  auto gridder = make_grid(traj, osamp.Get(), kernel.Get(), fastgrid, log);
  R2 const w = SDC::Choose(sdc.Get(), traj, osamp.Get(), log);
  gridder->setSDC(w);
  Cropper cropper(info, gridder->mapping().cartDims, out_fov.Get(), log);
  auto const cropSz = cropper.size();
  R3 const apo = gridder->apodization(cropSz);
  Cx4 sense;
  if (!rss) {
    if (senseFile) {
      sense = LoadSENSE(senseFile.Get(), log);
    } else {
      sense = DirectSENSE(
        info,
        gridder.get(),
        out_fov.Get(),
        senseLambda.Get(),
        reader.noncartesian(ValOrLast(senseVol.Get(), info.volumes)),
        log);
    }
  }

  if (basisFile) {
    HD5::Reader basisReader(basisFile.Get(), log);
    R2 const basis = basisReader.readBasis();
    gridder = make_grid_basis(gridder->mapping(), kernel.Get(), fastgrid, basis, log);
    gridder->setSDC(w);
  }
  gridder->setSDCPower(sdcPow.Get());

  Cx5 grid(gridder->inputDimensions(info.channels));
  Cx4 image(grid.dimension(1), cropSz[0], cropSz[1], cropSz[2]);
  Cx5 out(grid.dimension(1), cropSz[0], cropSz[1], cropSz[2], info.volumes);
  FFT::Planned<5, 3> fft(grid, log);
  auto dev = Threads::GlobalDevice();
  auto const &all_start = log.now();
  for (Index iv = 0; iv < info.volumes; iv++) {
    auto const &vol_start = log.now();
    gridder->Adj(reader.noncartesian(iv), grid);
    log.info("FFT...");
    fft.reverse(grid);
    log.info("Channel combination...");
    if (rss) {
      image.device(dev) = ConjugateSum(cropper.crop5(grid), cropper.crop5(grid)).sqrt();
    } else {
      image.device(dev) = ConjugateSum(
        cropper.crop5(grid),
        sense.reshape(Sz5{info.channels, 1, cropSz[0], cropSz[1], cropSz[2]})
          .broadcast(Sz5{1, grid.dimension(1), 1, 1, 1}));
    }
    image.device(dev) = image / apo.cast<Cx>()
                                  .reshape(Sz4{1, cropSz[0], cropSz[1], cropSz[2]})
                                  .broadcast(Sz4{grid.dimension(1), 1, 1, 1});
    out.chip<4>(iv) = image;
    log.info("Volume {}: {}", iv, log.toNow(vol_start));
  }
  log.info("All volumes: {}", log.toNow(all_start));
  auto const fname = OutName(iname.Get(), oname.Get(), "recon", "h5");
  HD5::Writer writer(fname, log);
  writer.writeInfo(info);
  writer.writeTensor(out, "image");
  FFT::End(log);
  return EXIT_SUCCESS;
}
