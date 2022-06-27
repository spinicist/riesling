#include "types.h"

#include "cropper.h"
#include "filter.h"
#include "log.h"
#include "op/recon.hpp"
#include "parse_args.h"
#include "sdc.h"
#include "sense.h"
#include "tensorOps.h"
#include "tgv.hpp"

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
  Trajectory const traj = reader.trajectory();
  auto const &info = traj.info();
  auto const kernel = make_kernel(core.ktype.Get(), info.type, core.osamp.Get());
  Mapping const mapping(reader.trajectory(), kernel.get(), core.osamp.Get(), core.bucketSize.Get());
  auto gridder = make_grid(kernel.get(), mapping, info.channels, core.basisFile.Get());
  auto const sdc = SDC::Choose(sdcOpts, traj, core.osamp.Get());
  Cx4 senseMaps = SENSE::Choose(senseOpts, info, gridder.get(), extra.iter_fov.Get(), sdc.get(), reader);
  ReconOp recon(gridder.get(), senseMaps, sdc.get());

  auto sz = recon.inputDimensions();
  Cropper out_cropper(info, LastN<3>(sz), extra.out_fov.Get());
  Sz3 outSz = out_cropper.size();
  Cx5 out(sz[0], outSz[0], outSz[1], outSz[2], info.volumes);
  auto const &all_start = Log::Now();
  for (Index iv = 0; iv < info.volumes; iv++) {
    auto const &vol_start = Log::Now();
    out.chip<4>(iv) = out_cropper.crop4(
      tgv(its.Get(), thr.Get(), alpha.Get(), reduce.Get(), step_size.Get(), recon, reader.noncartesian(iv)));
    Log::Print(FMT_STRING("Volume {}: {}"), iv, Log::ToNow(vol_start));
  }
  Log::Print(FMT_STRING("All Volumes: {}"), Log::ToNow(all_start));
  auto const fname = OutName(core.iname.Get(), core.oname.Get(), "tgv", "h5");
  HD5::Writer writer(fname);
  writer.writeInfo(info);
  writer.writeTensor(out, "image");

  return EXIT_SUCCESS;
}
