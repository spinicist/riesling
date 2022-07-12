#include "types.h"

#include "cropper.h"
#include "io/hd5.hpp"
#include "log.h"
#include "op/recon-rss.hpp"
#include "op/recon.hpp"
#include "parse_args.h"
#include "sdc.h"
#include "sense.h"
#include "tensorOps.h"

#include <variant>

using namespace rl;

int main_recon(args::Subparser &parser)
{
  CoreOpts core(parser);
  ExtraOpts extra(parser);
  SDC::Opts sdcOpts(parser);
  SENSE::Opts senseOpts(parser);
  args::Flag rss(parser, "RSS", "Use Root-Sum-Squares channel combination", {"rss", 'r'});
  args::Flag fwd(parser, "", "Apply forward operation", {'f', "fwd"});
  args::ValueFlag<std::string> trajName(parser, "T", "Override trajectory", {"traj"});
  args::ValueFlag<std::string> basisFile(parser, "BASIS", "Read subspace basis from .h5 file", {"basis", 'b'});

  ParseCommand(parser, core.iname);

  HD5::RieslingReader reader(core.iname.Get());
  Trajectory traj;
  if (trajName) {
    HD5::RieslingReader trajReader(trajName.Get());
    traj = trajReader.trajectory();
  } else {
    traj = reader.trajectory();
  }
  Info const &info = traj.info();

  auto const kernel = rl::make_kernel(core.ktype.Get(), info.type, core.osamp.Get());
  Mapping const mapping(reader.trajectory(), kernel.get(), core.osamp.Get(), core.bucketSize.Get());
  auto gridder = make_grid<Cx>(kernel.get(), mapping, info.channels, core.basisFile.Get());
  auto const sdc = SDC::Choose(sdcOpts, traj, core.osamp.Get());

  std::variant<nullptr_t, ReconOp, ReconRSSOp> recon = nullptr;
  Sz4 sz;
  if (rss) {
    if (fwd) {
      Log::Fail("RSS is not compatible with forward Recon Op");
    }
    Cropper crop(info, gridder->mapping().cartDims, extra.iter_fov.Get()); // To get correct dims
    recon.emplace<ReconRSSOp>(gridder.get(), crop.size(), sdc.get());
    sz = std::get<ReconRSSOp>(recon).inputDimensions();
  } else {
    Cx4 senseMaps = SENSE::Choose(senseOpts, info, gridder.get(), extra.iter_fov.Get(), sdc.get(), reader);
    recon.emplace<ReconOp>(gridder.get(), senseMaps, sdc.get());
    sz = std::get<ReconOp>(recon).inputDimensions();
  }
  Cropper out_cropper(info, LastN<3>(sz), extra.out_fov.Get());
  Sz3 outSz = out_cropper.size();

  if (fwd) {
    Log::Debug(FMT_STRING("Starting forward reconstruction op"));
    auto const &all_start = Log::Now();
    Cx5 images(sz[0], outSz[0], outSz[1], outSz[2], info.volumes);
    reader.readTensor(HD5::Keys::Image, images);
    Cx4 padded(sz);
    Cx4 kspace(info.channels, info.read_points, info.spokes, info.volumes);
    for (Index iv = 0; iv < info.volumes; iv++) {
      padded.setZero();
      out_cropper.crop4(padded) = images.chip<4>(iv);
      kspace.chip<3>(iv) = std::get<ReconOp>(recon).A(padded);
    }
    Log::Print(FMT_STRING("All Volumes: {}"), Log::ToNow(all_start));
    auto const fname = OutName(core.iname.Get(), core.oname.Get(), "recon", "h5");
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
    WriteOutput(out, core.iname.Get(), core.oname.Get(), parser.GetCommand().Name(), core.keepTrajectory, traj);
  }

  return EXIT_SUCCESS;
}
