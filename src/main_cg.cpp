#include "types.h"

#include "cg.hpp"
#include "filter.h"
#include "log.h"
#include "op/grid.h"
#include "op/recon.hpp"
#include "parse_args.h"
#include "sdc.h"
#include "sense.h"
#include "tensorOps.h"
#include "threads.h"

int main_cg(args::Subparser &parser)
{
  COMMON_RECON_ARGS;
  COMMON_SENSE_ARGS;
  args::Flag toeplitz(parser, "T", "Use Töplitz embedding", {"toe", 't'});
  args::ValueFlag<float> iter_fov(parser, "F", "Iterations FoV (default 256mm)", {"iter_fov"}, 256);
  args::ValueFlag<std::string> basisFile(parser, "BASIS", "Read basis from file", {"basis", 'b'});
  args::ValueFlag<float> thr(parser, "T", "Threshold for termination (1e-10)", {"thresh"}, 1.e-10);
  args::ValueFlag<Index> its(parser, "N", "Max iterations (8)", {'i', "max_its"}, 8);

  ParseCommand(parser, iname);
  FFT::Start();
  HD5::RieslingReader reader(iname.Get());
  Trajectory const traj = reader.trajectory();
  Info const &info = traj.info();

  auto const kernel = make_kernel(ktype.Get(), info.type, osamp.Get());
  auto const mapping = traj.mapping(kernel->inPlane(), osamp.Get());
  auto gridder = make_grid(kernel.get(), mapping, fastgrid);
  R2 const w = SDC::Choose(sdc.Get(), traj, osamp.Get());
  gridder->setSDC(w);
  Cx4 senseMaps = senseFile ? LoadSENSE(senseFile.Get())
                            : DirectSENSE(
                                info,
                                gridder.get(),
                                iter_fov.Get(),
                                senseLambda.Get(),
                                reader.noncartesian(ValOrLast(senseVol.Get(), info.volumes)));

  if (basisFile) {
    HD5::Reader basisReader(basisFile.Get());
    R2 const basis = basisReader.readTensor<R2>(HD5::Keys::Basis);
    gridder = make_grid_basis(kernel.get(), gridder->mapping(), basis, fastgrid);
    gridder->setSDC(w);
  }
  gridder->setSDCPower(sdcPow.Get());
  ReconOp recon(gridder.get(), senseMaps);
  if (toeplitz) {
    recon.calcToeplitz();
  }
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
    cg(its.Get(), thr.Get(), recon, vol);
    cropped = out_cropper.crop4(vol);
    out.chip<4>(iv) = cropped;
    Log::Print("Volume {}: {}", iv, Log::ToNow(vol_start));
  }
  Log::Print("All Volumes: {}", Log::ToNow(all_start));
  auto const fname = OutName(iname.Get(), oname.Get(), "cg", "h5");
  HD5::Writer writer(fname);
  writer.writeTrajectory(traj);
  writer.writeTensor(out, "image");
  FFT::End();
  return EXIT_SUCCESS;
}
