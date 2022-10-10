#include "types.hpp"

#include "algo/cg.hpp"
#include "cropper.h"
#include "log.hpp"
#include "op/recon.hpp"
#include "parse_args.hpp"
#include "sdc.hpp"
#include "sense.hpp"
#include "tensorOps.hpp"
#include "threads.hpp"

using namespace rl;

int main_cg(args::Subparser &parser)
{
  CoreOpts coreOpts(parser);
  SDC::Opts sdcOpts(parser);
  SENSE::Opts senseOpts(parser);
  args::Flag toeplitz(parser, "T", "Use Töplitz embedding", {"toe", 't'});
  args::ValueFlag<float> thr(parser, "T", "Termination threshold (1e-10)", {"thresh"}, 1.e-10);
  args::ValueFlag<Index> its(parser, "N", "Max iterations (8)", {"max-its"}, 8);

  ParseCommand(parser, coreOpts.iname);

  HD5::Reader reader(coreOpts.iname.Get());
  Trajectory traj(reader);
  Info const &info = traj.info();
  Index const channels = reader.dimensions<5>(HD5::Keys::Noncartesian)[0];
  Index const volumes = reader.dimensions<5>(HD5::Keys::Noncartesian)[4];
  auto const basis = ReadBasis(coreOpts.basisFile);
  auto const sdc = SDC::make_sdc(sdcOpts, traj, channels, coreOpts.ktype.Get(), coreOpts.osamp.Get());
  Cx4 senseMaps = SENSE::Choose(senseOpts, coreOpts, sdcOpts, traj, reader);
  ReconOp recon(traj, coreOpts.ktype.Get(), coreOpts.osamp.Get(), senseMaps, sdc.get(), basis, toeplitz);
  NormalEqOp<ReconOp> normEqs{recon};
  ConjugateGradients<NormalEqOp<ReconOp>> cg{normEqs, its.Get(), thr.Get(), true};

  auto sz = recon.inputDimensions();
  Cropper out_cropper(info.matrix, LastN<3>(sz), info.voxel_size, coreOpts.fov.Get());
  Cx4 vol(sz);
  Sz3 outSz = out_cropper.size();
  Cx4 cropped(sz[0], outSz[0], outSz[1], outSz[2]);
  Cx5 out(sz[0], outSz[0], outSz[1], outSz[2], volumes);
  auto const &all_start = Log::Now();
  for (Index iv = 0; iv < volumes; iv++) {
    auto const &vol_start = Log::Now();
    vol = cg.run(recon.adjoint(reader.readSlab<Cx4>(HD5::Keys::Noncartesian, iv)));
    cropped = out_cropper.crop4(vol);
    out.chip<4>(iv) = cropped;
    Log::Print(FMT_STRING("Volume {}: {}"), iv, Log::ToNow(vol_start));
  }
  Log::Print(FMT_STRING("All Volumes: {}"), Log::ToNow(all_start));
  WriteOutput(out, coreOpts.iname.Get(), coreOpts.oname.Get(), parser.GetCommand().Name(), coreOpts.keepTrajectory, traj);
  return EXIT_SUCCESS;
}
