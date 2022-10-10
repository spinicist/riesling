#include "types.hpp"

#include "cropper.h"
#include "io/hd5.hpp"
#include "log.hpp"
#include "op/recon-rss.hpp"
#include "parse_args.hpp"
#include "sdc.hpp"
#include "tensorOps.hpp"

using namespace rl;

int main_rss(args::Subparser &parser)
{
  CoreOpts coreOpts(parser);
  SDC::Opts sdcOpts(parser);
  args::ValueFlag<std::string> basisFile(parser, "BASIS", "Read subspace basis from .h5 file", {"basis", 'b'});

  ParseCommand(parser, coreOpts.iname);

  HD5::Reader reader(coreOpts.iname.Get());
  Trajectory traj(reader);
  Info const &info = traj.info();
  auto const basis = ReadBasis(coreOpts.basisFile);
  Index const channels = reader.dimensions<5>(HD5::Keys::Noncartesian)[0];
  Index const volumes = reader.dimensions<5>(HD5::Keys::Noncartesian)[3];
  auto const sdc = SDC::make_sdc(sdcOpts, traj, channels, coreOpts.ktype.Get(), coreOpts.osamp.Get());
  ReconRSSOp recon(traj, coreOpts.ktype.Get(), coreOpts.osamp.Get(), channels, sdc.get(), basis);
  Sz4 sz = recon.inputDimensions();
  Cropper out_cropper(info.matrix, LastN<3>(sz), info.voxel_size, coreOpts.fov.Get());
  Sz3 outSz = out_cropper.size();

  Cx5 out(sz[0], outSz[0], outSz[1], outSz[2], volumes);
  auto const &all_start = Log::Now();
  for (Index iv = 0; iv < volumes; iv++) {
    out.chip<4>(iv) = out_cropper.crop4(recon.adjoint(reader.readSlab<Cx4>(HD5::Keys::Noncartesian, iv)));
  }
  Log::Print(FMT_STRING("All Volumes: {}"), Log::ToNow(all_start));
  WriteOutput(out, coreOpts.iname.Get(), coreOpts.oname.Get(), parser.GetCommand().Name(), coreOpts.keepTrajectory, traj);

  return EXIT_SUCCESS;
}
