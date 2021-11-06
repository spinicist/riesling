#include "types.h"

#include "cropper.h"
#include "fft_plan.hpp"
#include "filter.h"
#include "log.h"
#include "op/grid-basis.h"
#include "parse_args.h"
#include "sense.h"
#include "tensorOps.h"

int main_basis_recon(args::Subparser &parser)
{
  COMMON_RECON_ARGS;
  COMMON_SENSE_ARGS;

  args::Flag rss(parser, "RSS", "Use Root-Sum-Squares channel combination", {"rss", 'r'});
  args::Flag save_channels(
    parser, "CHANNELS", "Write out individual channels from first volume", {"channels", 'c'});
  args::ValueFlag<std::string> basisFile(
    parser, "BASIS", "Read subspace basis from .h5 file", {"basis", 'b'});

  Log log = ParseCommand(parser, iname);
  FFT::Start(log);
  HD5::Reader reader(iname.Get(), log);
  auto const traj = reader.readTrajectory();
  auto const &info = traj.info();

  HD5::Reader basisReader(basisFile.Get(), log);
  R2 const basis = basisReader.readBasis();
  long const nB = basis.dimension(1);

  auto gridder = make_grid_basis(traj, osamp.Get(), kb, fastgrid, basis, log);
  auto grid1 = make_grid(gridder->mapping(), kb, fastgrid, log);
  SDC::Choose(sdc.Get(), traj, grid1, gridder, log);
  gridder->setSDCExponent(sdc_exp.Get());
  Cropper cropper(info, gridder->gridDims(), out_fov.Get(), log);
  Sz3 const cropSz = cropper.size();
  R3 const apo = gridder->apodization(cropper.size());
  auto const gridSz = gridder->gridDims();
  Cx5 grid(info.channels, nB, gridSz[0], gridSz[1], gridSz[2]);
  Cx4 images(nB, cropSz[0], cropSz[1], cropSz[2]);
  Cx5 out(nB, cropSz[0], cropSz[1], cropSz[2], info.volumes);
  out.setZero();
  images.setZero();
  FFT::ThreeDBasis fft(grid, log);

  Cx4 sense = rss ? Cx4() : cropper.newMultichannel(info.channels);
  if (!rss) {
    if (senseFile) {
      sense = LoadSENSE(senseFile.Get(), log);
    } else {
      sense = DirectSENSE(
        traj, osamp.Get(), kb, out_fov.Get(), senseLambda.Get(), senseVol.Get(), reader, log);
    }
  }

  auto dev = Threads::GlobalDevice();
  auto const &all_start = log.now();
  for (long iv = 0; iv < info.volumes; iv++) {
    auto const &vol_start = log.now();
    grid.setZero();
    gridder->Adj(reader.noncartesian(iv), grid);
    log.info("FFT...");
    fft.reverse(grid);
    log.info("Channel combination...");
    if (rss) {
      images.device(dev) = ConjugateSum(cropper.crop5(grid), cropper.crop5(grid)).sqrt();
    } else {
      images.device(dev) = ConjugateSum(
        cropper.crop5(grid),
        sense.reshape(Sz5{info.channels, 1, cropSz[0], cropSz[1], cropSz[2]})
          .broadcast(Sz5{1, nB, 1, 1, 1}));
    }
    images.device(dev) =
      images /
      apo.cast<Cx>().reshape(Sz4{1, cropSz[0], cropSz[1], cropSz[2]}).broadcast(Sz4{nB, 1, 1, 1});
    out.chip(iv, 4) = images;
    log.info("Volume {}: {}", iv, log.toNow(vol_start));
  }
  log.info("All volumes: {}", log.toNow(all_start));
  WriteBasisVolumes(
    out, basis, mag, info, iname.Get(), oname.Get(), "basis-recon", oftype.Get(), log);
  FFT::End(log);
  return EXIT_SUCCESS;
}
