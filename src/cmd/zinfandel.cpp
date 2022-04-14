#include "io/io.h"
#include "log.h"
#include "parse_args.h"
#include "threads.h"
#include "types.h"
#include "zin-grappa.hpp"
#include "zin-slr.hpp"
#include <filesystem>

int main_zinfandel(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "INPUT FILE", "Input radial k-space to fill");
  args::ValueFlag<std::string> oname(
    parser, "OUTPUT NAME", "Name of output .h5 file", {"out", 'o'});

  args::ValueFlag<Index> gap(parser, "G", "Set gap value (default 2)", {'g', "gap"}, 2);

  args::Flag grappa(parser, "", "Use projection GRAPPA", {"grappa"});
  args::ValueFlag<Index> gSrc(parser, "S", "GRAPPA sources (default 4)", {"grappa-src"}, 4);
  args::ValueFlag<Index> gSpokes(
    parser, "S", "GRAPPA calibration spokes (default 4)", {"spokes"}, 4);
  args::ValueFlag<Index> gRead(
    parser, "R", "GRAPPA calibration read samples (default 8)", {"read"}, 8);
  args::ValueFlag<float> gλ(parser, "L", "Tikhonov regularization (default 0)", {"lamda"}, 0.f);

  // SLR options
  args::ValueFlag<float> osamp(parser, "OS", "Grid oversampling factor (2)", {'s', "os"}, 2.f);
  args::ValueFlag<std::string> ktype(
    parser, "K", "Choose kernel - NN, KB3, KB5", {'k', "kernel"}, "KB3");
  args::ValueFlag<float> res(parser, "R", "Resolution for SLR (default 5mm)", {'r', "res"}, 5.f);

  args::ValueFlag<Index> inner_its(parser, "ITS", "Max inner iterations (2)", {"max-its"}, 2);
  args::ValueFlag<Index> outer_its(parser, "ITS", "Max outer iterations (8)", {"max-outer-its"}, 8);
  args::ValueFlag<float> reg_rho(parser, "R", "ADMM rho (default 0.1)", {"rho"}, 0.1f);
  args::Flag precond(parser, "P", "Apply Ong's single-channel M-conditioner", {"pre"});
  args::ValueFlag<float> atol(parser, "A", "Tolerance on A", {"atol"}, 1.e-6f);
  args::ValueFlag<float> btol(parser, "B", "Tolerance on b", {"btol"}, 1.e-6f);
  args::ValueFlag<float> ctol(parser, "C", "Tolerance on cond(A)", {"ctol"}, 1.e-6f);

  args::ValueFlag<float> lambda(
    parser, "L", "Regularization parameter (default 0.1)", {"lambda"}, 0.1f);
  args::ValueFlag<Index> patchSize(parser, "SZ", "Patch size (default 4)", {"patch-size"}, 4);

  ParseCommand(parser, iname);

  HD5::RieslingReader reader(iname.Get());
  auto const traj = reader.trajectory();
  auto info = traj.info();
  auto out_info = info;

  Cx4 rad_ks = info.noncartesianSeries();

  if (grappa) {
    for (Index iv = 0; iv < info.volumes; iv++) {
      Cx3 vol = reader.noncartesian(iv);
      zinGRAPPA(gap.Get(), gSrc.Get(), gSpokes.Get(), gRead.Get(), gλ.Get(), traj.points(), vol);
      rad_ks.chip<3>(iv) = vol;
    }
  } else {
    // Use SLR
    auto const kernel = make_kernel(ktype.Get(), info.type, osamp.Get());
    auto const mapping = traj.mapping(kernel->inPlane(), osamp.Get(), 0, res.Get(), true);
    auto gridder = make_grid(kernel.get(), mapping, false);

    for (Index iv = 0; iv < info.volumes; iv++) {
      Cx3 vol = reader.noncartesian(iv);
      // zinSLR(gap.Get(), gridder.get(), vol);
      rad_ks.chip<3>(iv) = vol;
    }
  }

  HD5::Writer writer(OutName(iname.Get(), oname.Get(), "zinfandel", "h5"));
  writer.writeTrajectory(Trajectory(out_info, traj.points()));
  writer.writeMeta(reader.readMeta());
  writer.writeTensor(rad_ks, HD5::Keys::Noncartesian);
  Log::Print(FMT_STRING("Finished"));
  return EXIT_SUCCESS;
}
