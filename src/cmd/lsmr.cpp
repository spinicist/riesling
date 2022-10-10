#include "types.hpp"

#include "algo/lsmr.hpp"
#include "cropper.h"
#include "log.hpp"
#include "op/recon.hpp"
#include "parse_args.hpp"
#include "func/pre-kspace.hpp"
#include "sdc.hpp"
#include "sense.hpp"
#include "tensorOps.hpp"
#include "threads.hpp"

using namespace rl;

int main_lsmr(args::Subparser &parser)
{
  CoreOpts coreOpts(parser);
  SDC::Opts sdcOpts(parser);
  SENSE::Opts senseOpts(parser);
  args::ValueFlag<Index> its(parser, "N", "Max iterations (8)", {'i', "max-its"}, 8);
  args::Flag pre(parser, "P", "Apply Ong's single-channel pre-conditioner", {"pre"});
  args::ValueFlag<float> atol(parser, "A", "Tolerance on A (1e-6)", {"atol"}, 1.e-6f);
  args::ValueFlag<float> btol(parser, "B", "Tolerance on b (1e-6)", {"btol"}, 1.e-6f);
  args::ValueFlag<float> ctol(parser, "C", "Tolerance on cond(A) (1e-6)", {"ctol"}, 1.e-6f);
  args::ValueFlag<float> λ(parser, "λ", "Tikhonov parameter (default 0)", {"lambda"}, 0.f);

  ParseCommand(parser, coreOpts.iname);

  HD5::Reader reader(coreOpts.iname.Get());
  Trajectory traj(reader);
  Info const &info = traj.info();
  auto const basis = ReadBasis(coreOpts.basisFile);
  Index const channels = reader.dimensions<5>(HD5::Keys::Noncartesian)[0];
  Index const volumes = reader.dimensions<5>(HD5::Keys::Noncartesian)[4];
  auto const sdc = SDC::make_sdc(sdcOpts, traj, channels, coreOpts.ktype.Get(), coreOpts.osamp.Get());
  Cx4 senseMaps = SENSE::Choose(senseOpts, coreOpts, sdcOpts, traj, reader);
  ReconOp recon(traj, coreOpts.ktype.Get(), coreOpts.osamp.Get(), senseMaps, sdc.get(), basis);
  std::unique_ptr<Functor<Cx4>> M = pre ? std::make_unique<KSpaceSingle>(traj) : nullptr;
  LSMR<ReconOp> lsmr{recon, M.get(), its.Get(), atol.Get(), btol.Get(), ctol.Get(), true};
  auto sz = recon.inputDimensions();
  Cropper out_cropper(info.matrix, LastN<3>(sz), info.voxel_size, coreOpts.fov.Get());
  Cx4 vol(sz);
  Sz3 outSz = out_cropper.size();
  Cx4 cropped(sz[0], outSz[0], outSz[1], outSz[2]);
  Cx5 out(sz[0], outSz[0], outSz[1], outSz[2], volumes);

  auto const &all_start = Log::Now();
  for (Index iv = 0; iv < volumes; iv++) {
    auto const &vol_start = Log::Now();
    vol = lsmr.run(reader.readSlab<Cx4>(HD5::Keys::Noncartesian, iv), λ.Get());
    cropped = out_cropper.crop4(vol);
    out.chip<4>(iv) = cropped;
    Log::Print(FMT_STRING("Volume {}: {}"), iv, Log::ToNow(vol_start));
  }
  Log::Print(FMT_STRING("All Volumes: {}"), Log::ToNow(all_start));
  WriteOutput(out, coreOpts.iname.Get(), coreOpts.oname.Get(), parser.GetCommand().Name(), coreOpts.keepTrajectory, traj);
  return EXIT_SUCCESS;
}
