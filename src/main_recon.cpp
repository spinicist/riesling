#include "types.h"

#include "cropper.h"
#include "fft_plan.h"
#include "filter.h"
#include "io.h"
#include "log.h"
#include "op/grid-basis.h"
#include "op/grid.h"
#include "parse_args.h"
#include "sense.h"
#include "tensorOps.h"

int main_recon(args::Subparser &parser)
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
  auto gridder = make_grid(traj, osamp.Get(), kb, fastgrid, log);
  R2 const w = SDC::Choose(sdc.Get(), traj, gridder, log);
  gridder->setSDC(w);
  Cropper cropper(info, gridder->gridDims(), out_fov.Get(), log);
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
    long const nB = basis.dimension(1);
    auto basisGridder = make_grid_basis(gridder->mapping(), kb, fastgrid, basis, log);
    gridder->setSDC(w);
    auto const gridSz = gridder->gridDims();
    Cx5 grid(info.channels, nB, gridSz[0], gridSz[1], gridSz[2]);
    Cx4 images(nB, cropSz[0], cropSz[1], cropSz[2]);
    Cx5 out(nB, cropSz[0], cropSz[1], cropSz[2], info.volumes);
    out.setZero();
    images.setZero();
    FFT::ThreeDBasis fft(grid, log);

    auto dev = Threads::GlobalDevice();
    auto const &all_start = log.now();
    for (long iv = 0; iv < info.volumes; iv++) {
      auto const &vol_start = log.now();
      grid.setZero();
      basisGridder->Adj(reader.noncartesian(iv), grid);
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
  } else {

    Cx4 grid = gridder->newMultichannel(info.channels);
    Cx3 image = cropper.newImage();
    Cx4 out = cropper.newSeries(info.volumes);
    out.setZero();
    image.setZero();
    FFT::ThreeDMulti fft(grid, log);

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
        image.device(dev) = ConjugateSum(cropper.crop4(grid), cropper.crop4(grid)).sqrt();
      } else {
        image.device(dev) = ConjugateSum(cropper.crop4(grid), sense);
      }
      image.device(dev) = image / apo.cast<Cx>();
      if (tukey_s || tukey_e || tukey_h) {
        ImageTukey(tukey_s.Get(), tukey_e.Get(), tukey_h.Get(), image, log);
      }
      out.chip(iv, 3) = image;
      log.info("Volume {}: {}", iv, log.toNow(vol_start));
      if (save_channels && (iv == 0)) {
        Cx4 const cropped = FirstToLast4(cropper.crop4(grid));
        WriteOutput(
          cropped, false, false, info, iname.Get(), oname.Get(), "channels", oftype.Get(), log);
      }
    }
    log.info("All volumes: {}", log.toNow(all_start));
    WriteOutput(out, mag, false, info, iname.Get(), oname.Get(), "recon", oftype.Get(), log);
  }
  FFT::End(log);
  return EXIT_SUCCESS;
}
