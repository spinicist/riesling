#include "sdc.hpp"

#include "io/hd5.hpp"
#include "log.hpp"
#include "mapping.hpp"
#include "op/tensorop.hpp"
#include "op/make_grid.hpp"
#include "op/scale.hpp"
#include "tensorOps.hpp"
#include "threads.hpp"
#include "trajectory.hpp"

namespace rl {

namespace SDC {

Opts::Opts(args::Subparser &parser, std::string const &def)
  : type(parser, "SDC", "SDC type: 'pipe', 'none', or filename", {"sdc"}, def)
  , pow(parser, "P", "SDC Power (default 1.0)", {"sdc-pow"}, 1.0f)
  , maxIterations(parser, "I", "SDC Max iterations (40)", {"sdc-its"}, 40)
{
}

template <int ND>
auto Pipe(Trajectory const &traj, std::string const &ktype, float const os, Index const its, float const pow) -> Re2
{
  Log::Print("Using Pipe/Zwart/Menon SDC...");
  Re3 W(1, traj.nSamples(), traj.nTraces());
  Re3 Wp(W.dimensions());
  auto gridder = make_grid<float, ND>(traj, ktype, os, 1);

  W.setConstant(1.f);
  for (Index ii = 0; ii < its; ii++) {
    Wp = gridder->forward(gridder->adjoint(W));
    Wp.device(Threads::GlobalDevice()) = (Wp > 0.f).select(W / Wp, Wp.constant(0.f)).eval(); // Avoid divide by zero problems
    float const delta = Norm(Wp - W) / Norm(W);
    W.device(Threads::GlobalDevice()) = Wp;
    if (delta < 1e-6) {
      Log::Print("SDC converged, delta was {}", delta);
      break;
    } else {
      Log::Print("SDC Step {}/{} Delta {}", ii, its, delta);
    }
  }
  Log::Print("SDC finished.");
  return W.chip<0>(0).pow(pow);
}

Re2 Radial2D(Trajectory const &traj)
{
  Log::Print("Calculating 2D radial analytic SDC");
  Info const &info = traj.info();
  auto spoke_sdc = [&](Index const spoke, Index const N) -> Re1 {
    float const k_delta = Norm(traj.point(1, spoke) - traj.point(0, spoke));
    float const V = 2.f * k_delta * M_PI / N; // Area element
    // When k-space becomes undersampled need to flatten DC (Menon & Pipe 1999)
    float const R = (M_PI * *std::max_element(info.matrix.begin(), info.matrix.end())) / N;
    float const flat_start = traj.nSamples() / sqrt(R);
    float const flat_val = V * flat_start;
    Re1 sdc(traj.nSamples());
    for (Index ir = 0; ir < traj.nSamples(); ir++) {
      float const rad = traj.nSamples() * Norm(traj.point(ir, spoke));
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

  Re1 const ss = spoke_sdc(0, traj.nTraces());
  Re2 sdc = ss.reshape(Sz2{traj.nSamples(), 1}).broadcast(Sz2{1, traj.nTraces()});
  return sdc;
}

Re2 Radial3D(Trajectory const &traj, Index const lores, Index const gap)
{
  Log::Print("Calculating 2D radial analytic SDC");
  auto const &info = traj.info();

  Eigen::ArrayXf ind = Eigen::ArrayXf::LinSpaced(traj.nSamples(), 0, traj.nSamples() - 1);
  Eigen::ArrayXf mergeHi = ind - (gap - 1);
  mergeHi = (mergeHi > 0).select(mergeHi, 0);
  mergeHi = (mergeHi < 1).select(mergeHi, 1);

  Eigen::ArrayXf mergeLo;
  if (lores) {
    float const scale = Norm(traj.point(traj.nSamples() - 1, lores)) / Norm(traj.point(traj.nSamples() - 1, 0));
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
    float const flat_start = traj.nSamples() / R;
    float const V = 1.f / (3. * (flat_start * flat_start) + 1. / 4.);
    Re1 sdc(traj.nSamples());
    for (Index ir = 0; ir < traj.nSamples(); ir++) {
      float const rad = traj.nSamples() * Norm(traj.point(ir, spoke));
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

  Re2 sdc(traj.nSamples(), traj.nTraces());
  if (lores) {
    Re1 const ss = spoke_sdc(0, lores);
    sdc.slice(Sz2{0, 0}, Sz2{traj.nSamples(), lores}) = ss.reshape(Sz2{traj.nSamples(), 1}).broadcast(Sz2{1, lores});
  }
  Re1 const ss = spoke_sdc(lores, traj.nTraces() - lores);
  sdc.slice(Sz2{0, lores}, Sz2{traj.nSamples(), traj.nTraces() - lores}) =
    ss.reshape(Sz2{traj.nSamples(), 1}).broadcast(Sz2{1, traj.nTraces() - lores});

  return sdc;
}

Re2 Radial(Trajectory const &traj, Index const lores, Index const gap) { return Radial3D(traj, lores, gap); }

auto Choose(SDC::Opts &opts, Index const nC, Trajectory const &traj, std::string const &ktype, float const os)
  -> std::shared_ptr<TensorOperator<Cx, 3>>
{
  Re2 sdc(traj.nSamples(), traj.nTraces());
  auto const iname = opts.type.Get();
  Sz3 const dims{nC, traj.nSamples(), traj.nTraces()};
  if (iname == "" || iname == "none") {
    Log::Print("Using no density compensation");
    return std::make_shared<TensorIdentity<Cx, 3>>(dims);
  } else if (iname == "pipe") {
    if (traj.nDims() == 2) {
      sdc = SDC::Pipe<2>(traj, ktype, os, opts.maxIterations.Get(), opts.pow.Get());
    } else {
      sdc = SDC::Pipe<3>(traj, ktype, os, opts.maxIterations.Get(), opts.pow.Get());
    }
  } else {
    HD5::Reader reader(iname);
    sdc = reader.readTensor<Re2>(HD5::Keys::SDC);
    if (sdc.dimension(0) != traj.nSamples() || sdc.dimension(1) != traj.nTraces()) {
      Log::Fail(
        "SDC dimensions on disk {}x{} did not match info {}x{}",
        sdc.dimension(0),
        sdc.dimension(1),
        traj.nSamples(),
        traj.nTraces());
    }
  }
  return std::make_shared<Scale<Cx, 3>>(dims, sdc.cast<Cx>());
}

} // namespace SDC
} // namespace rl
