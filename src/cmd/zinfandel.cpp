#include "algo/admm.hpp"
#include "algo/lsqr.hpp"
#include "io/hd5.hpp"
#include "log.h"
#include "op/nufft.hpp"
#include "parse_args.h"
#include "precond/single.hpp"
#include "threads.h"
#include "types.h"
#include "zin-grappa.hpp"
#include "zin-slr.hpp"
#include <filesystem>

using namespace rl;

int main_zinfandel(args::Subparser &parser)
{
  CoreOpts core(parser);

  args::ValueFlag<Index> gap(parser, "G", "Set gap value (default 2)", {'g', "gap"}, 2);

  args::Flag grappa(parser, "", "Use projection GRAPPA", {"grappa"});
  args::ValueFlag<Index> gSrc(parser, "S", "GRAPPA sources (default 4)", {"grappa-src"}, 4);
  args::ValueFlag<Index> gtraces(parser, "S", "GRAPPA calibration traces (default 4)", {"traces"}, 4);
  args::ValueFlag<Index> gRead(parser, "R", "GRAPPA calibration read samples (default 8)", {"read"}, 8);
  args::ValueFlag<float> gλ(parser, "L", "Tikhonov regularization (default 0)", {"lamda"}, 0.f);

  // SLR options
  args::ValueFlag<float> res(parser, "R", "Resolution for SLR (default 20mm)", {'r', "res"}, 20.f);

  args::ValueFlag<Index> iits(parser, "ITS", "Max inner iterations (4)", {"max-its"}, 4);
  args::ValueFlag<Index> oits(parser, "ITS", "Max outer iterations (8)", {"max-outer-its"}, 16);
  args::ValueFlag<float> rho(parser, "R", "ADMM rho (default 0.1)", {"rho"}, 1.0f);
  args::ValueFlag<float> atol(parser, "A", "Tolerance on A", {"atol"}, 1.e-4f);
  args::ValueFlag<float> btol(parser, "B", "Tolerance on b", {"btol"}, 1.e-4f);
  args::ValueFlag<float> ctol(parser, "C", "Tolerance on cond(A)", {"ctol"}, 1.e-6f);
  args::ValueFlag<float> winSz(parser, "T", "SLR normalized window size (default 1.5)", {"win-size"}, 0.1f);
  args::ValueFlag<Index> kSz(parser, "SZ", "SLR Kernel Size (default 4)", {"kernel-size"}, 4);

  ParseCommand(parser, core.iname);

  HD5::RieslingReader reader(core.iname.Get());
  auto const traj = reader.trajectory();
  auto info = traj.info();
  auto out_info = info;

  Cx4 rad_ks = info.noncartesianSeries();

  if (grappa) {
    for (Index iv = 0; iv < info.volumes; iv++) {
      Cx3 vol = reader.noncartesian(iv);
      zinGRAPPA(gap.Get(), gSrc.Get(), gtraces.Get(), gRead.Get(), gλ.Get(), traj.points(), vol);
      rad_ks.chip<3>(iv) = vol;
    }
  } else {
    // Use SLR
    auto const [dsTraj, minRead] = traj.downsample(res.Get(), 0, true);
    auto const dsInfo = dsTraj.info();
    
    auto grid0 = make_grid<Cx, 3>(dsTraj, core.ktype.Get(), core.osamp.Get(), info.channels);
    auto gridN = make_grid<Cx, 3>(dsTraj, core.ktype.Get(), core.osamp.Get(), info.channels);

    NUFFTOp nufft0(LastN<3>(grid0->inputDimensions()), grid0.get());
    NUFFTOp nufftN(LastN<3>(gridN->inputDimensions()), gridN.get());
    auto const pre = std::make_unique<SingleChannel>(dsTraj);
    auto reg = [&](Cx5 const &x) -> Cx5 { return zinSLR(x, nufftN.fft(), kSz.Get(), winSz.Get()); };
    Sz3 const st{0, 0, 0};
    Sz3 const sz{info.channels, gap.Get(), info.traces};
    LSQR<NUFFTOp> lsqr{nufftN, pre.get(), nullptr, iits.Get(), atol.Get(), btol.Get(), ctol.Get(), rho.Get(), false};
    ADMM<LSQR<NUFFTOp>> admm{lsqr, reg, oits.Get(), rho.Get()};
    for (Index iv = 0; iv < info.volumes; iv++) {
      Cx3 const ks = reader.noncartesian(iv);
      Cx3 const dsKS = ks.slice(Sz3{0, minRead, 0}, Sz3{dsInfo.channels, dsInfo.samples, dsInfo.traces});
      Cx5 const img = admm.run(dsKS);
      Cx3 const filled = nufft0.forward(img);
      rad_ks.chip<3>(iv) = ks;
      rad_ks.chip<3>(iv).slice(st, sz) = filled.slice(st, sz);
    }
  }

  HD5::Writer writer(OutName(core.iname.Get(), core.oname.Get(), "zinfandel", "h5"));
  writer.writeTrajectory(Trajectory(out_info, traj.points()));
  writer.writeMeta(reader.readMeta());
  writer.writeTensor(rad_ks, HD5::Keys::Noncartesian);
  Log::Print(FMT_STRING("Finished"));
  return EXIT_SUCCESS;
}
