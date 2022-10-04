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
  CoreOpts core(parser);
  ExtraOpts extra(parser);
  SDC::Opts sdcOpts(parser);
  args::ValueFlag<std::string> basisFile(parser, "BASIS", "Read subspace basis from .h5 file", {"basis", 'b'});

  ParseCommand(parser, core.iname);

  HD5::Reader reader(core.iname.Get());
  Trajectory traj(reader);
  Info const &info = traj.info();
  auto const basis = ReadBasis(core.basisFile);
  Index const channels = reader.dimensions<4>(HD5::Keys::Noncartesian)[0];
  Index const volumes = reader.dimensions<4>(HD5::Keys::Noncartesian)[3];
  auto gridder = make_grid<Cx, 3>(traj, core.ktype.Get(), core.osamp.Get(), channels, basis);
  auto const sdc = SDC::Choose(sdcOpts, traj, channels, core.ktype.Get(), core.osamp.Get());
  ReconRSSOp recon(gridder.get(), LastN<3>(info.matrix), sdc.get());
  Sz4 sz = recon.inputDimensions();
  Cropper out_cropper(info.matrix, LastN<3>(sz), info.voxel_size, extra.out_fov.Get());
  Sz3 outSz = out_cropper.size();

  Cx5 out(sz[0], outSz[0], outSz[1], outSz[2], volumes);
  auto const &all_start = Log::Now();
  for (Index iv = 0; iv < volumes; iv++) {
    out.chip<4>(iv) = out_cropper.crop4(recon.adjoint(reader.readSlab<Cx3>(HD5::Keys::Noncartesian, iv)));
  }
  Log::Print(FMT_STRING("All Volumes: {}"), Log::ToNow(all_start));
  WriteOutput(out, core.iname.Get(), core.oname.Get(), parser.GetCommand().Name(), core.keepTrajectory, traj);

  return EXIT_SUCCESS;
}
