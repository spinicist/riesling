#include "sdc.h"

#include "io/hd5.hpp"
#include "mapping.hpp"
#include "op/gridBase.hpp"
#include "op/sdc.hpp"
#include "tensorOps.hpp"
#include "threads.hpp"
#include "trajectory.hpp"

namespace rl {

namespace SDC {

Opts::Opts(args::Subparser &parser)
  : type(parser, "SDC", "SDC type: 'pipe', 'pipenn', 'none', or filename", {"sdc"}, "pipe")
  , pow(parser, "P", "SDC Power (default 1.0)", {"sdc-pow"}, 1.0f)
  , maxIterations(parser, "I", "SDC Max iterations (40)", {"sdc-its"}, 40)
{
}

Re2 Pipe(Trajectory const &inTraj, std::string const &ktype, float const os, Index const its)
{
  Log::Print(FMT_STRING("Using Pipe/Zwart/Menon SDC..."));
  auto info = inTraj.info();
  Trajectory traj{info, inTraj.points(), inTraj.frames()};
  Re3 W(1, info.samples, info.traces);
  Re3 Wp(W.dimensions());
  std::unique_ptr<GridBase<float, 3>> gridder = make_grid<float, 3>(traj, ktype, os, 1);

  W.setConstant(1.f);
  for (Index ii = 0; ii < its; ii++) {
    Wp = gridder->forward(gridder->adjoint(W));
    Wp.device(Threads::GlobalDevice()) =
      (Wp > 0.f).select(W / Wp, Wp.constant(0.f)).eval(); // Avoid divide by zero problems
    float const delta = Norm(Wp - W) / Norm(W);
    W.device(Threads::GlobalDevice()) = Wp;
    if (delta < 1e-7) {
      Log::Print(FMT_STRING("SDC converged, delta was {}"), delta);
      break;
    } else {
      Log::Print(FMT_STRING("SDC Step {}/{} Delta {}"), ii, its, delta);
    }
  }
  Log::Print(FMT_STRING("SDC finished."));
  return W.chip<0>(0);
}

Re2 Radial2D(Trajectory const &traj)
{
  Log::Print(FMT_STRING("Calculating 2D radial analytic SDC"));
  Info const &info = traj.info();
  auto spoke_sdc = [&](Index const spoke, Index const N) -> Re1 {
    float const k_delta = Norm(traj.point(1, spoke) - traj.point(0, spoke));
    float const V = 2.f * k_delta * M_PI / N; // Area element
    // When k-space becomes undersampled need to flatten DC (Menon & Pipe 1999)
    float const R = (M_PI * *std::max_element(info.matrix.begin(), info.matrix.end())) / N;
    float const flat_start = info.samples / sqrt(R);
    float const flat_val = V * flat_start;
    Re1 sdc(info.samples);
    for (Index ir = 0; ir < info.samples; ir++) {
      float const rad = info.samples * Norm(traj.point(ir, spoke));
      if (rad == 0.f) {
        sdc(ir) = V / 8.f;
      } else if (rad < flat_start) {
        sdc(ir) = V * rad;
      } else {
        sdc(ir) = flat_val;
      }
    }
    return sdc;
  };

  Re1 const ss = spoke_sdc(0, info.traces);
  Re2 sdc = ss.reshape(Sz2{info.samples, 1}).broadcast(Sz2{1, info.traces});
  return sdc;
}

Re2 Radial3D(Trajectory const &traj, Index const lores, Index const gap)
{
  Log::Print(FMT_STRING("Calculating 2D radial analytic SDC"));
  auto const &info = traj.info();

  Eigen::ArrayXf ind = Eigen::ArrayXf::LinSpaced(info.samples, 0, info.samples - 1);
  Eigen::ArrayXf mergeHi = ind - (gap - 1);
  mergeHi = (mergeHi > 0).select(mergeHi, 0);
  mergeHi = (mergeHi < 1).select(mergeHi, 1);

  Eigen::ArrayXf mergeLo;
  if (lores) {
    float const scale = Norm(traj.point(info.samples - 1, lores)) / Norm(traj.point(info.samples - 1, 0));
    mergeLo = ind / scale - (gap - 1);
    mergeLo = (mergeLo > 0).select(mergeLo, 0);
    mergeLo = (mergeLo < 1).select(mergeLo, 1);
    mergeLo = (1 - mergeLo) / scale; // Match intensities of k-space
    mergeLo.head(gap) = 0.;          // Don't touch these points
  }

  auto spoke_sdc = [&](Index const &spoke, Index const N) -> Re1 {
    // When k-space becomes undersampled need to flatten DC (Menon & Pipe 1999)
    auto const mm = *std::max_element(info.matrix.begin(), info.matrix.end());
    float const R = (M_PI * mm * mm) / N;
    float const flat_start = info.samples / R;
    float const V = 1.f / (3. * (flat_start * flat_start) + 1. / 4.);
    Re1 sdc(info.samples);
    for (Index ir = 0; ir < info.samples; ir++) {
      float const rad = info.samples * Norm(traj.point(ir, spoke));
      float const merge = (spoke < lores) ? mergeLo(ir) : mergeHi(ir);
      if (rad == 0.f) {
        sdc(ir) = merge * V * 1.f / 8.f;
      } else if (rad < flat_start) {
        sdc(ir) = merge * V * (3.f * (rad * rad) + 1.f / 4.f);
      } else {
        sdc(ir) = merge;
      }
    }
    return sdc;
  };

  Re2 sdc(info.samples, info.traces);
  if (lores) {
    Re1 const ss = spoke_sdc(0, lores);
    sdc.slice(Sz2{0, 0}, Sz2{info.samples, lores}) = ss.reshape(Sz2{info.samples, 1}).broadcast(Sz2{1, lores});
  }
  Re1 const ss = spoke_sdc(lores, info.traces - lores);
  sdc.slice(Sz2{0, lores}, Sz2{info.samples, info.traces - lores}) =
    ss.reshape(Sz2{info.samples, 1}).broadcast(Sz2{1, info.traces - lores});

  return sdc;
}

Re2 Radial(Trajectory const &traj, Index const lores, Index const gap)
{

  if (traj.info().grid3D()) {
    return Radial3D(traj, lores, gap);
  } else {
    return Radial2D(traj);
  }
}

std::unique_ptr<SDCOp> Choose(Opts &opts, Trajectory const &traj, std::string const &ktype, float const os)
{
  Re2 sdc(traj.info().samples, traj.info().traces);
  auto const iname = opts.type.Get();
  if (iname == "" || iname == "none") {
    Log::Print(FMT_STRING("Using no density compensation"));
    auto const info = traj.info();
    return std::make_unique<SDCOp>(Sz2{info.samples, info.traces}, info.channels);
  } else if (iname == "pipe") {
    sdc = Pipe(traj, ktype, os, opts.maxIterations.Get());
  } else {
    HD5::Reader reader(iname);
    sdc = reader.readTensor<Re2>(HD5::Keys::SDC);
    auto const trajInfo = traj.info();
    if (sdc.dimension(0) != trajInfo.samples || sdc.dimension(1) != trajInfo.traces) {
      Log::Fail(
        FMT_STRING("SDC dimensions on disk {}x{} did not match info {}x{}"),
        sdc.dimension(0),
        sdc.dimension(1),
        trajInfo.samples,
        trajInfo.traces);
    }
  }
  return std::make_unique<SDCOp>(sdc.pow(opts.pow.Get()), traj.info().channels);
}

} // namespace SDC
} // namespace rl
