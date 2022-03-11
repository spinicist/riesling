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
  COMMON_RECON_ARGS;
  COMMON_SENSE_ARGS;

  args::ValueFlag<float> thr(
    parser, "TRESHOLD", "Threshold for termination (1e-10)", {"thresh"}, 1.e-10);
  args::ValueFlag<Index> its(
    parser, "MAX ITS", "Maximum number of iterations (16)", {'i', "max_its"}, 16);
  args::ValueFlag<std::string> basisFile(
    parser, "BASIS", "Read subspace basis from .h5 file", {"basis", 'b'});
  args::ValueFlag<float> alpha(
    parser, "ALPHA", "Regularisation weighting (1e-5)", {"alpha"}, 1.e-5f);
  args::ValueFlag<float> reduce(
    parser, "REDUCE", "Reduce regularisation over iters (suggest 0.1)", {"reduce"}, 1.f);
  args::ValueFlag<float> step_size(
    parser, "STEP SIZE", "Inverse of step size (default 8)", {"step"}, 8.f);

  ParseCommand(parser, iname);

  HD5::RieslingReader reader(iname.Get());
  Trajectory const traj = reader.trajectory();
  auto const &info = traj.info();
  auto const kernel = make_kernel(ktype.Get(), info.type, osamp.Get());
  auto const mapping = traj.mapping(kernel->inPlane(), osamp.Get());
  auto gridder = make_grid(kernel.get(), mapping, fastgrid);
  auto const sdc = SDC::Choose(sdcType.Get(), sdcPow.Get(), traj, osamp.Get());
  Cx4 senseMaps = sFile ? LoadSENSE(sFile.Get())
                        : SelfCalibration(
                            info,
                            gridder.get(),
                            iter_fov.Get(),
                            sRes.Get(),
                            sReg.Get(),
                            sdc->apply(reader.noncartesian(ValOrLast(sVol.Get(), info.volumes))));

  if (basisFile) {
    HD5::Reader basisReader(basisFile.Get());
    R2 const basis = basisReader.readTensor<R2>(HD5::Keys::Basis);
    gridder = make_grid_basis(kernel.get(), gridder->mapping(), basis, fastgrid);
  }
  ReconOp recon(gridder.get(), senseMaps, sdc.get());

  auto sz = recon.inputDimensions();
  Cropper out_cropper(info, LastN<3>(sz), out_fov.Get());
  Sz3 outSz = out_cropper.size();
  Cx5 out(sz[0], outSz[0], outSz[1], outSz[2], info.volumes);
  auto const &all_start = Log::Now();
  for (Index iv = 0; iv < info.volumes; iv++) {
    auto const &vol_start = Log::Now();
    out.chip<4>(iv) = out_cropper.crop4(tgv(
      its.Get(),
      thr.Get(),
      alpha.Get(),
      reduce.Get(),
      step_size.Get(),
      recon,
      reader.noncartesian(iv)));
    Log::Print(FMT_STRING("Volume {}: {}"), iv, Log::ToNow(vol_start));
  }
  Log::Print(FMT_STRING("All Volumes: {}"), Log::ToNow(all_start));
  auto const fname = OutName(iname.Get(), oname.Get(), "tgv", "h5");
  HD5::Writer writer(fname);
  writer.writeInfo(info);
  writer.writeTensor(out, "image");

  return EXIT_SUCCESS;
}
