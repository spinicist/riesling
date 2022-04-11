#include "types.h"

#include "cropper.h"
#include "io/io.h"
#include "log.h"
#include "op/recon-rss.hpp"
#include "op/recon.hpp"
#include "parse_args.h"
#include "sdc.h"
#include "sense.h"
#include "tensorOps.h"

#include <variant>

int main_recon(args::Subparser &parser)
{
  COMMON_RECON_ARGS;
  COMMON_SENSE_ARGS;
  args::Flag rss(parser, "RSS", "Use Root-Sum-Squares channel combination", {"rss", 'r'});
  args::Flag fwd(parser, "F", "Apply forward operation", {"fwd"});
  args::ValueFlag<std::string> trajName(parser, "T", "Override trajectory", {"traj"});
  args::ValueFlag<std::string> basisFile(
    parser, "BASIS", "Read subspace basis from .h5 file", {"basis", 'b'});

  ParseCommand(parser, iname);

  HD5::RieslingReader reader(iname.Get());
  Trajectory traj;
  if (trajName) {
    HD5::RieslingReader trajReader(trajName.Get());
    traj = trajReader.trajectory();
  } else {
    traj = reader.trajectory();
  }
  Info const &info = traj.info();

  auto const kernel = make_kernel(ktype.Get(), info.type, osamp.Get());
  auto const mapping = traj.mapping(kernel->inPlane(), osamp.Get());
  auto gridder = make_grid(kernel.get(), mapping, fastgrid);
  auto const sdc = SDC::Choose(sdcType.Get(), traj, osamp.Get(), sdcPow.Get());
  std::unique_ptr<GridBase> bgridder = nullptr;
  if (basisFile) {
    HD5::Reader basisReader(basisFile.Get());
    R2 const basis = basisReader.readTensor<R2>(HD5::Keys::Basis);
    bgridder = make_grid_basis(kernel.get(), mapping, basis, fastgrid);
  }

  std::variant<nullptr_t, ReconOp, ReconRSSOp> recon = nullptr;

  Sz4 sz;
  if (rss) {
    if (fwd) {
      Log::Fail("RSS is not compatible with forward Recon Op");
    }
    Cropper crop(info, gridder->mapping().cartDims, iter_fov.Get()); // To get correct dims
    recon.emplace<ReconRSSOp>(basisFile ? bgridder.get() : gridder.get(), crop.size(), sdc.get());
    sz = std::get<ReconRSSOp>(recon).inputDimensions();
  } else {
    Cx4 senseMaps = SENSE::Choose(
      sFile.Get(),
      info,
      gridder.get(),
      iter_fov.Get(),
      sRes.Get(),
      sReg.Get(),
      sdc.get(),
      reader.noncartesian(ValOrLast(sVol.Get(), info.volumes)));
    recon.emplace<ReconOp>(basisFile ? bgridder.get() : gridder.get(), senseMaps, sdc.get());
    sz = std::get<ReconOp>(recon).inputDimensions();
  }
  Cropper out_cropper(info, LastN<3>(sz), out_fov.Get());
  Sz3 outSz = out_cropper.size();

  if (fwd) {
    Log::Debug(FMT_STRING("Starting forward reconstruction op"));
    auto const &all_start = Log::Now();
    Cx5 images(sz[0], outSz[0], outSz[1], outSz[2], info.volumes);
    reader.readTensor(HD5::Keys::Image, images);
    Cx4 padded(sz);
    Cx4 kspace(info.channels, info.read_points, info.spokes, info.volumes);
    for (Index iv = 0; iv < info.volumes; iv++) {
      out_cropper.crop4(padded) = images.chip<4>(iv);
      kspace.chip<3>(iv) = std::get<ReconOp>(recon).A(padded);
    }
    Log::Print(FMT_STRING("All Volumes: {}"), Log::ToNow(all_start));
    auto const fname = OutName(iname.Get(), oname.Get(), "recon", "h5");
    HD5::Writer writer(fname);
    writer.writeTrajectory(traj);
    writer.writeTensor(kspace, HD5::Keys::Noncartesian);
  } else {
    Cx4 vol(sz);
    Cx4 cropped(sz[0], outSz[0], outSz[1], outSz[2]);
    Cx5 out(sz[0], outSz[0], outSz[1], outSz[2], info.volumes);
    auto const &all_start = Log::Now();
    for (Index iv = 0; iv < info.volumes; iv++) {
      if (rss) {
        vol = std::get<ReconRSSOp>(recon).Adj(reader.noncartesian(iv));
      } else {
        vol = std::get<ReconOp>(recon).Adj(reader.noncartesian(iv));
      }
      cropped = out_cropper.crop4(vol);
      out.chip<4>(iv) = cropped;
    }
    Log::Print(FMT_STRING("All Volumes: {}"), Log::ToNow(all_start));
    auto const fname = OutName(iname.Get(), oname.Get(), "recon", "h5");
    HD5::Writer writer(fname);
    writer.writeTrajectory(traj);
    writer.writeTensor(out, HD5::Keys::Image);
  }

  return EXIT_SUCCESS;
}
