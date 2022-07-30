#include "types.h"

#include "algo/lsqr.hpp"
#include "cropper.h"
#include "log.h"
#include "op/recon-sense.hpp"
#include "parse_args.h"
#include "precond/scaling.hpp"
#include "precond/single.hpp"
#include "sdc.h"
#include "sense.h"
#include "tensorOps.h"
#include "threads.h"

using namespace rl;

int main_lsqr(args::Subparser &parser)
{
  CoreOpts core(parser);
  ExtraOpts extra(parser);
  SDC::Opts sdcOpts(parser);
  SENSE::Opts senseOpts(parser);
  args::ValueFlag<Index> its(parser, "N", "Max iterations (8)", {'i', "max-its"}, 8);
  args::ValueFlag<Index> readStart(parser, "N", "Read start", {"rs"}, 0);
  args::Flag lp(parser, "M", "Apply Ong's single-channel pre-conditioner", {"lpre"});
  args::Flag rp(parser, "N", "Apply right preconditioner (scales)", {"rpre"});
  args::ValueFlag<float> atol(parser, "A", "Tolerance on A (1e-6)", {"atol"}, 1.e-6f);
  args::ValueFlag<float> btol(parser, "B", "Tolerance on b (1e-6)", {"btol"}, 1.e-6f);
  args::ValueFlag<float> ctol(parser, "C", "Tolerance on cond(A) (1e-6)", {"ctol"}, 1.e-6f);
  args::ValueFlag<float> damp(parser, "λ", "Tikhonov parameter (default 0)", {"lambda"}, 0.f);

  ParseCommand(parser, core.iname);

  HD5::RieslingReader reader(core.iname.Get());
  Trajectory const traj = reader.trajectory();
  Info const &info = traj.info();

  auto const kernel = rl::make_kernel(core.ktype.Get(), info.type, core.osamp.Get());
  Mapping const mapping(reader.trajectory(), kernel.get(), core.osamp.Get(), core.bucketSize.Get());
  auto gridder = make_grid<Cx>(kernel.get(), mapping, info.channels, core.basisFile.Get());
  auto const sdc = SDC::Choose(sdcOpts, traj, core.osamp.Get());
  Cx4 senseMaps = SENSE::Choose(senseOpts, info, gridder.get(), extra.iter_fov.Get(), sdc.get(), reader);
  auto recon = std::make_unique<ReconSENSE>(gridder.get(), senseMaps, sdc.get(), false);
  auto const sz = recon->inputDimensions();

  std::unique_ptr<Precond<Cx3>> M = lp ? std::make_unique<SingleChannel>(traj, kernel.get()) : nullptr;
  std::unique_ptr<Precond<Cx4>> N = nullptr;
  if (rp) {
    auto sgrid = make_grid<Cx>(kernel.get(), mapping, 1, core.basisFile.Get());
    NUFFTOp snufft(LastN<3>(sz), sgrid.get());
    Cx3 ones(sgrid->outputDimensions());
    ones.setConstant(1.f);
    auto const &imgs = snufft.Adj(ones);
    R1 scales = (imgs.conjugate() * imgs).real().sum(Sz4{0, 2, 3, 4}).sqrt();
    scales /= scales.constant(scales[0]);
    N = std::make_unique<Scaling>(sz, scales);
  }

  Cropper out_cropper(info, LastN<3>(sz), extra.out_fov.Get());
  Cx4 vol(sz);
  Sz3 outSz = out_cropper.size();
  Cx4 cropped(sz[0], outSz[0], outSz[1], outSz[2]);
  Cx5 out(sz[0], outSz[0], outSz[1], outSz[2], info.volumes);

  auto const &all_start = Log::Now();
  for (Index iv = 0; iv < info.volumes; iv++) {
    auto const &vol_start = Log::Now();
    vol = lsqr(
      its.Get(),
      recon.get(),
      reader.noncartesian(iv),
      atol.Get(),
      btol.Get(),
      ctol.Get(),
      damp.Get(),
      M.get(),
      N.get(),
      true);
    cropped = out_cropper.crop4(vol);
    out.chip<4>(iv) = cropped;
    Log::Print(FMT_STRING("Volume {}: {}"), iv, Log::ToNow(vol_start));
  }
  Log::Print(FMT_STRING("All Volumes: {}"), Log::ToNow(all_start));
  WriteOutput(out, core.iname.Get(), core.oname.Get(), parser.GetCommand().Name(), core.keepTrajectory, traj);
  return EXIT_SUCCESS;
}
