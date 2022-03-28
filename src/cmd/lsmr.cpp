#include "types.h"

#include "algo/lsmr.hpp"
#include "log.h"
#include "op/recon.hpp"
#include "parse_args.h"
#include "precond/single.hpp"
#include "sdc.h"
#include "sense.h"
#include "tensorOps.h"
#include "threads.h"

int main_lsmr(args::Subparser &parser)
{
  COMMON_RECON_ARGS;
  COMMON_SENSE_ARGS;
  args::ValueFlag<std::string> basisFile(parser, "BASIS", "Read basis from file", {"basis", 'b'});
  args::ValueFlag<Index> its(parser, "N", "Max iterations (8)", {'i', "max-its"}, 8);
  args::Flag precond(parser, "P", "Apply Ong's single-channel pre-conditioner", {"pre"});
  args::ValueFlag<float> atol(parser, "A", "Tolerance on A", {"atol"}, 1.e-6f);
  args::ValueFlag<float> btol(parser, "B", "Tolerance on b", {"btol"}, 1.e-6f);
  args::ValueFlag<float> ctol(parser, "C", "Tolerance on cond(A)", {"ctol"}, 1.e-6f);
  args::ValueFlag<float> damp(parser, "D", "Damping (regularization) factor", {"damp"}, 0.f);

  ParseCommand(parser, iname);

  HD5::RieslingReader reader(iname.Get());
  Trajectory const traj = reader.trajectory();
  Info const &info = traj.info();

  auto const kernel = make_kernel(ktype.Get(), info.type, osamp.Get());
  auto const mapping = traj.mapping(kernel->inPlane(), osamp.Get());
  auto gridder = make_grid(kernel.get(), mapping, fastgrid);

  std::unique_ptr<Precond> pre =
    precond.Get() ? std::make_unique<SingleChannel>(gridder.get()) : nullptr;
  Cx4 senseMaps = sFile ? LoadSENSE(sFile.Get())
                        : SelfCalibration(
                            info,
                            gridder.get(),
                            iter_fov.Get(),
                            sRes.Get(),
                            sReg.Get(),
                            SDC::Choose(sdcType.Get(), sdcPow.Get(), traj, osamp.Get())
                              ->apply(reader.noncartesian(ValOrLast(sVol.Get(), info.volumes))));

  if (basisFile) {
    HD5::Reader basisReader(basisFile.Get());
    R2 const basis = basisReader.readTensor<R2>(HD5::Keys::Basis);
    gridder = make_grid_basis(kernel.get(), gridder->mapping(), basis, fastgrid);
  }
  ReconOp recon(gridder.get(), senseMaps);
  auto sz = recon.inputDimensions();
  Cropper out_cropper(info, LastN<3>(sz), out_fov.Get());
  Cx4 vol(sz);
  Sz3 outSz = out_cropper.size();
  Cx4 cropped(sz[0], outSz[0], outSz[1], outSz[2]);
  Cx5 out(sz[0], outSz[0], outSz[1], outSz[2], info.volumes);
  auto const &all_start = Log::Now();
  for (Index iv = 0; iv < info.volumes; iv++) {
    auto const &vol_start = Log::Now();
    vol = lsmr(
      its.Get(),
      recon,
      reader.noncartesian(iv),
      pre.get(),
      atol.Get(),
      btol.Get(),
      ctol.Get(),
      damp.Get());
    cropped = out_cropper.crop4(vol);
    out.chip<4>(iv) = cropped;
    Log::Print(FMT_STRING("Volume {}: {}"), iv, Log::ToNow(vol_start));
  }
  Log::Print(FMT_STRING("All Volumes: {}"), Log::ToNow(all_start));
  auto const fname = OutName(iname.Get(), oname.Get(), "lsmr", "h5");
  HD5::Writer writer(fname);
  writer.writeTrajectory(traj);
  writer.writeTensor(out, "image");

  return EXIT_SUCCESS;
}
