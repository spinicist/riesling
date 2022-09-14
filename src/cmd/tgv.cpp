#include "types.h"

#include "cropper.h"
#include "log.hpp"
#include "op/recon.hpp"
#include "parse_args.hpp"
#include "sdc.hpp"
#include "sense.h"
#include "tensorOps.hpp"
#include "algo/tgv.hpp"

using namespace rl;

int main_tgv(args::Subparser &parser)
{
  CoreOpts core(parser);
  ExtraOpts extra(parser);
  SDC::Opts sdcOpts(parser);
  SENSE::Opts senseOpts(parser);
  args::ValueFlag<float> thr(parser, "TRESHOLD", "Threshold for termination (1e-10)", {"thresh"}, 1.e-10);
  args::ValueFlag<Index> its(parser, "MAX ITS", "Maximum number of iterations (16)", {'i', "max-its"}, 16);
  args::ValueFlag<float> alpha(parser, "ALPHA", "Regularisation weighting (1e-5)", {"alpha"}, 1.e-5f);
  args::ValueFlag<float> reduce(parser, "REDUCE", "Reduce regularisation over iters (suggest 0.1)", {"reduce"}, 1.f);
  args::ValueFlag<float> step_size(parser, "STEP SIZE", "Inverse of step size (default 8)", {"step"}, 8.f);
  ParseCommand(parser, core.iname);

  HD5::RieslingReader reader(core.iname.Get());
  auto const traj = reader.trajectory();
  Info const &info = traj.info();
  auto const basis = ReadBasis(core.basisFile);
  Index const channels = reader.dimensions<4>(HD5::Keys::Noncartesian)[0];
  Index const volumes = reader.dimensions<4>(HD5::Keys::Noncartesian)[3];
  auto gridder = make_grid<Cx, 3>(traj, core.ktype.Get(), core.osamp.Get(), channels, basis);
  auto const sdc = SDC::Choose(sdcOpts, traj, channels, core.ktype.Get(), core.osamp.Get());
  Cx4 senseMaps = SENSE::Choose(senseOpts, traj, gridder.get(), extra.iter_fov.Get(), sdc.get(), reader);
  ReconOp recon(gridder.get(), senseMaps, sdc.get());

  auto sz = recon.inputDimensions();
  Cropper out_cropper(info.matrix, LastN<3>(sz), info.voxel_size, extra.out_fov.Get());
  Sz3 outSz = out_cropper.size();
  Cx5 out(sz[0], outSz[0], outSz[1], outSz[2], volumes);
  auto const &all_start = Log::Now();
  for (Index iv = 0; iv < volumes; iv++) {
    auto const &vol_start = Log::Now();
    out.chip<4>(iv) = out_cropper.crop4(
      tgv(its.Get(), thr.Get(), alpha.Get(), reduce.Get(), step_size.Get(), recon, reader.noncartesian(iv)));
    Log::Print(FMT_STRING("Volume {}: {}"), iv, Log::ToNow(vol_start));
  }
  Log::Print(FMT_STRING("All Volumes: {}"), Log::ToNow(all_start));
  WriteOutput(out, core.iname.Get(), core.oname.Get(), parser.GetCommand().Name(), core.keepTrajectory, traj);
  return EXIT_SUCCESS;
}
