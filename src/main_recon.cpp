#include "types.h"

#include "cropper.h"
#include "fft_plan.h"
#include "filter.h"
#include "io.h"
#include "log.h"
#include "op/grid.h"
#include "parse_args.h"
#include "sdc.h"
#include "sense.h"
#include "tensorOps.h"

int main_recon(args::Subparser &parser)
{
  COMMON_RECON_ARGS;
  COMMON_SENSE_ARGS;
  args::Flag rss(parser, "RSS", "Use Root-Sum-Squares channel combination", {"rss", 'r'});
  args::ValueFlag<std::string> basisFile(
    parser, "BASIS", "Read subspace basis from .h5 file", {"basis", 'b'});

  ParseCommand(parser, iname);
  FFT::Start();
  HD5::RieslingReader reader(iname.Get());
  auto const traj = reader.trajectory();
  auto const &info = traj.info();
  auto const kernel = make_kernel(ktype.Get(), info.type, osamp.Get());
  auto const mapping = traj.mapping(kernel->inPlane(), osamp.Get());
  auto gridder = make_grid(kernel.get(), mapping, fastgrid);
  R2 const w = SDC::Choose(sdc.Get(), traj, osamp.Get());
  gridder->setSDC(w);
  Cropper cropper(info, gridder->mapping().cartDims, out_fov.Get());
  auto const cropSz = cropper.size();
  R3 const apo = gridder->apodization(cropSz);
  Cx4 sense;
  if (!rss) {
    if (senseFile) {
      sense = LoadSENSE(senseFile.Get());
      if (
        sense.dimension(0) != info.channels || sense.dimension(1) != cropSz[0] ||
        sense.dimension(2) != cropSz[1] || sense.dimension(3) != cropSz[2]) {
        Log::Fail(
          FMT_STRING("SENSE dimensions on disk were {}, expected {},{}"),
          fmt::join(sense.dimensions(), ","),
          info.channels,
          fmt::join(cropSz, ","));
      }
    } else {
      sense = DirectSENSE(
        info,
        gridder.get(),
        out_fov.Get(),
        senseLambda.Get(),
        reader.noncartesian(ValOrLast(senseVol.Get(), info.volumes)));
    }
  }

  if (basisFile) {
    HD5::Reader basisReader(basisFile.Get());
    R2 const basis = basisReader.readTensor<R2>(HD5::Keys::Basis);
    gridder = make_grid_basis(kernel.get(), gridder->mapping(), basis, fastgrid);
    gridder->setSDC(w);
  }
  gridder->setSDCPower(sdcPow.Get());
  Cx4 image(gridder->inputDimensions()[1], cropSz[0], cropSz[1], cropSz[2]);
  Cx5 out(gridder->inputDimensions()[1], cropSz[0], cropSz[1], cropSz[2], info.volumes);
  FFT::Planned<5, 3> fft(gridder->inputDimensions());
  auto dev = Threads::GlobalDevice();
  auto const &all_start = Log::Now();
  for (Index iv = 0; iv < info.volumes; iv++) {
    auto const &vol_start = Log::Now();
    gridder->Adj(reader.noncartesian(iv));
    fft.reverse(gridder->workspace());
    Log::Print(FMT_STRING("Channel combination..."));
    if (rss) {
      image.device(dev) =
        ConjugateSum(cropper.crop5(gridder->workspace()), cropper.crop5(gridder->workspace()))
          .sqrt();
    } else {
      image.device(dev) = ConjugateSum(
        cropper.crop5(gridder->workspace()),
        sense.reshape(Sz5{info.channels, 1, cropSz[0], cropSz[1], cropSz[2]})
          .broadcast(Sz5{1, out.dimension(0), 1, 1, 1}));
    }
    image.device(dev) = image / apo.cast<Cx>()
                                  .reshape(Sz4{1, cropSz[0], cropSz[1], cropSz[2]})
                                  .broadcast(Sz4{out.dimension(0), 1, 1, 1});
    out.chip<4>(iv) = image;
    Log::Print(FMT_STRING("Volume {}: {}"), iv, Log::ToNow(vol_start));
  }
  Log::Print(FMT_STRING("All volumes: {}"), Log::ToNow(all_start));
  auto const fname = OutName(iname.Get(), oname.Get(), "recon", "h5");
  HD5::Writer writer(fname);
  writer.writeInfo(info);
  writer.writeTensor(out, "image");
  FFT::End();
  return EXIT_SUCCESS;
}
