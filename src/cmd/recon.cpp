#include "types.hpp"

#include "cropper.h"
#include "io/hd5.hpp"
#include "log.hpp"
#include "op/recon.hpp"
#include "parse_args.hpp"
#include "sdc.hpp"
#include "sense.h"
#include "tensorOps.hpp"

using namespace rl;

int main_recon(args::Subparser &parser)
{
  CoreOpts core(parser);
  ExtraOpts extra(parser);
  SDC::Opts sdcOpts(parser);
  SENSE::Opts senseOpts(parser);
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
  auto const basis = ReadBasis(core.basisFile);

  Index volumes = fwd ? reader.dimensions<5>(HD5::Keys::Image)[4] : reader.dimensions<4>(HD5::Keys::Noncartesian)[3];

  if (fwd) {
    if (!senseOpts.file) {
      Log::Fail("Must specify SENSE maps for forward recon");
    }
    HD5::Reader senseReader(senseOpts.file.Get());
    Cx4 senseMaps = senseReader.readTensor<Cx4>(HD5::Keys::SENSE);
    Index channels = senseMaps.dimension(0);
    auto gridder = make_grid<Cx, 3>(traj, core.ktype.Get(), core.osamp.Get(), channels, basis);
    ReconOp recon(gridder.get(), senseMaps);
    Sz4 const sz = recon.inputDimensions();
    Cropper out_cropper(info.matrix, LastN<3>(recon.inputDimensions()), info.voxel_size, extra.out_fov.Get());

    auto const &all_start = Log::Now();
    auto const images = reader.readTensor<Cx5>(HD5::Keys::Image);
    Cx4 padded(sz);
    Cx4 kspace(channels, traj.nSamples(), traj.nTraces(), volumes);
    for (Index iv = 0; iv < volumes; iv++) {
      padded.setZero();
      out_cropper.crop4(padded) = images.chip<4>(iv);
      kspace.chip<3>(iv) = recon.forward(padded);
    }
    Log::Print(FMT_STRING("All Volumes: {}"), Log::ToNow(all_start));
    auto const fname = OutName(core.iname.Get(), core.oname.Get(), "recon", "h5");
    HD5::Writer writer(fname);
    writer.writeTrajectory(traj);
    writer.writeTensor(kspace, HD5::Keys::Noncartesian);
  } else {
    Index const channels = reader.dimensions<4>(HD5::Keys::Noncartesian)[0];
    auto gridder = make_grid<Cx, 3>(traj, core.ktype.Get(), core.osamp.Get(), channels, basis);
    std::unique_ptr<SDCOp> const sdc =
      fwd ? nullptr : SDC::Choose(sdcOpts, traj, channels, core.ktype.Get(), core.osamp.Get());
    Cx4 senseMaps = SENSE::Choose(senseOpts, traj, gridder.get(), extra.iter_fov.Get(), sdc.get(), reader);
    ReconOp recon(gridder.get(), senseMaps, sdc.get());
    Sz4 const sz = recon.inputDimensions();
    Cropper out_cropper(info.matrix, LastN<3>(recon.inputDimensions()), info.voxel_size, extra.out_fov.Get());
    Sz3 const outSz = out_cropper.size();
    Cx4 vol(sz);
    Cx4 cropped(sz[0], outSz[0], outSz[1], outSz[2]);
    Cx5 out(sz[0], outSz[0], outSz[1], outSz[2], volumes);
    auto const &all_start = Log::Now();
    for (Index iv = 0; iv < volumes; iv++) {
      vol = recon.adjoint(reader.noncartesian(iv));
      cropped = out_cropper.crop4(vol);
      out.chip<4>(iv) = cropped;
    }
    Log::Print(FMT_STRING("All Volumes: {}"), Log::ToNow(all_start));
    WriteOutput(out, core.iname.Get(), core.oname.Get(), parser.GetCommand().Name(), core.keepTrajectory, traj);
  }

  return EXIT_SUCCESS;
}
