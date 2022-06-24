#include "types.h"

#include "algo/cg.hpp"
#include "log.h"
#include "op/recon.hpp"
#include "parse_args.h"
#include "sdc.h"
#include "sense.h"
#include "tensorOps.h"
#include "threads.h"

int main_cg(args::Subparser &parser)
{
  CoreOpts core(parser);
  ExtraOpts extra(parser);
  SDC::Opts sdcOpts(parser);
  SENSE::Opts senseOpts(parser);
  args::Flag toeplitz(parser, "T", "Use Töplitz embedding", {"toe", 't'});
  args::ValueFlag<std::string> basisFile(parser, "BASIS", "Read basis from file", {"basis", 'b'});
  args::ValueFlag<float> thr(parser, "T", "Termination threshold (1e-10)", {"thresh"}, 1.e-10);
  args::ValueFlag<Index> its(parser, "N", "Max iterations (8)", {"max-its"}, 8);

  ParseCommand(parser, core.iname);

  HD5::RieslingReader reader(core.iname.Get());
  Trajectory const traj = reader.trajectory();
  Info const &info = traj.info();

  auto const kernel = make_kernel(core.ktype.Get(), info.type, core.osamp.Get());
  auto const mapping = traj.mapping(kernel->inPlane(), core.osamp.Get());
  auto gridder = make_grid(kernel.get(), mapping, info.channels, core.fast);
  auto const sdc = SDC::Choose(sdcOpts, traj, core.osamp.Get());
  Cx4 senseMaps = SENSE::Choose(senseOpts, info, gridder.get(), extra.iter_fov.Get(), sdc.get(), reader);

  if (basisFile) {
    HD5::Reader basisReader(basisFile.Get());
    R2 const basis = basisReader.readTensor<R2>(HD5::Keys::Basis);
    gridder = make_grid_basis(kernel.get(), gridder->mapping(), info.channels, basis, core.fast);
  }
  ReconOp recon(gridder.get(), senseMaps, sdc.get());
  if (toeplitz) {
    recon.calcToeplitz();
  }
  NormalEqOp<ReconOp> normEqs{recon};
  auto sz = recon.inputDimensions();
  Cropper out_cropper(info, LastN<3>(sz), extra.out_fov.Get());
  Cx4 vol(sz);
  Sz3 outSz = out_cropper.size();
  Cx4 cropped(sz[0], outSz[0], outSz[1], outSz[2]);
  Cx5 out(sz[0], outSz[0], outSz[1], outSz[2], info.volumes);
  auto const &all_start = Log::Now();
  for (Index iv = 0; iv < info.volumes; iv++) {
    auto const &vol_start = Log::Now();
    vol = cg(its.Get(), thr.Get(), normEqs, recon.Adj(reader.noncartesian(iv)));
    cropped = out_cropper.crop4(vol);
    out.chip<4>(iv) = cropped;
    Log::Print(FMT_STRING("Volume {}: {}"), iv, Log::ToNow(vol_start));
  }
  Log::Print(FMT_STRING("All Volumes: {}"), Log::ToNow(all_start));
  auto const fname = OutName(core.iname.Get(), core.oname.Get(), "cg", "h5");
  HD5::Writer writer(fname);
  writer.writeTrajectory(traj);
  writer.writeTensor(out, "image");

  return EXIT_SUCCESS;
}
