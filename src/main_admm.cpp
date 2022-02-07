#include "types.h"

#include "admm.hpp"
#include "cropper.h"
#include "fft_plan.h"
#include "filter.h"
#include "io.h"
#include "llr.h"
#include "log.h"
#include "op/grid.h"
#include "op/recon.hpp"
#include "parse_args.h"
#include "sdc.h"
#include "sense.h"

int main_admm(args::Subparser &parser)
{
  COMMON_RECON_ARGS;
  COMMON_SENSE_ARGS;
  args::Flag toeplitz(parser, "T", "Use Töplitz embedding", {"toe", 't'});
  args::ValueFlag<float> iter_fov(parser, "F", "Iterations FoV (default 256mm)", {"iter_fov"}, 256);
  args::ValueFlag<std::string> basisFile(parser, "BASIS", "Read basis from file", {"basis", 'b'});
  args::ValueFlag<float> cg_thr(parser, "T", "CG threshold (1e-10)", {"cg_thresh"}, 1.e-10);
  args::ValueFlag<Index> cg_its(parser, "ITS", "CG iterations (8)", {"cg_its"}, 8);
  args::ValueFlag<Index> admm_its(parser, "ITS", "Outer iterations (8)", {"admm_its"}, 8);
  args::ValueFlag<float> reg_lambda(parser, "L", "ADMM lambda (default 0.1)", {"reg"}, 0.1f);
  args::ValueFlag<float> reg_rho(parser, "R", "ADMM rho (default 0.1)", {"rho"}, 0.1f);
  args::ValueFlag<Index> patch(parser, "P", "Patch size for LLR (default 8)", {"patch"}, 8);

  ParseCommand(parser, iname);
  FFT::Start();

  HD5::RieslingReader reader(iname.Get());
  Trajectory const traj = reader.trajectory();
  Info const &info = traj.info();

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
                            sdc(reader.noncartesian(ValOrLast(sVol.Get(), info.volumes))));

  if (basisFile) {
    HD5::Reader basisReader(basisFile.Get());
    R2 const basis = basisReader.readTensor<R2>(HD5::Keys::Basis);
    gridder = make_grid_basis(kernel.get(), gridder->mapping(), basis, fastgrid);
  }
  ReconOp recon(gridder.get(), senseMaps);
  if (toeplitz) {
    recon.calcToeplitz(sdc);
  }
  auto reg = [&](Cx4 const &x) -> Cx4 { return llr(x, reg_lambda.Get(), patch.Get()); };

  auto sz = recon.inputDimensions();
  Cropper out_cropper(info, LastN<3>(sz), out_fov.Get());
  Cx4 vol(sz);
  Sz3 outSz = out_cropper.size();
  Cx4 cropped(sz[0], outSz[0], outSz[1], outSz[2]);
  Cx5 out(sz[0], outSz[0], outSz[1], outSz[2], info.volumes);
  auto const &all_start = Log::Now();
  for (Index iv = 0; iv < info.volumes; iv++) {
    auto const &vol_start = Log::Now();
    vol = recon.Adj(reader.noncartesian(iv)); // Initialize
    vol = admm(
      admm_its.Get(),
      cg_its.Get(),
      cg_thr.Get(),
      recon,
      sdc,
      reg,
      reg_rho.Get(),
      reader.noncartesian(iv));
    cropped = out_cropper.crop4(vol);
    out.chip<4>(iv) = cropped;
    Log::Print(FMT_STRING("Volume {}: {}"), iv, Log::ToNow(vol_start));
  }
  Log::Print(FMT_STRING("All Volumes: {}"), Log::ToNow(all_start));
  auto const fname = OutName(iname.Get(), oname.Get(), "admm", "h5");
  HD5::Writer writer(fname);
  writer.writeInfo(info);
  writer.writeTensor(out, "image");
  FFT::End();
  return EXIT_SUCCESS;
}
