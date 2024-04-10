#include "sdc.hpp"

#include "basis/basis.hpp"
#include "io/hd5.hpp"
#include "log.hpp"
#include "mapping.hpp"
#include "op/grid.hpp"
#include "op/tensorop.hpp"
#include "op/tensorscale.hpp"
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
  Re3  W(1, traj.nSamples(), traj.nTraces());
  Re3  Wp(W.dimensions());
  auto gridder = Grid<float, ND>::Make(traj, ktype, os, 1);

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
      Log::Print("SDC Step {}/{} Delta {}", ii + 1, its, delta);
    }
  }
  Log::Print("SDC finished.");
  return W.chip<0>(0).pow(pow);
}

Re2 Radial2D(Trajectory const &traj)
{
  Log::Print("Calculating 2D radial analytic SDC");
  auto spoke_sdc = [&](Index const spoke, Index const N) -> Re1 {
    float const k_delta = Norm(traj.point(1, spoke) - traj.point(0, spoke));
    float const V = 2.f * k_delta * M_PI / N; // Area element
    // When k-space becomes undersampled need to flatten DC (Menon & Pipe 1999)
    float const R = (M_PI * *std::max_element(traj.matrix().begin(), traj.matrix().end())) / N;
    float const flat_start = traj.nSamples() / sqrt(R);
    float const flat_val = V * flat_start;
    Re1         sdc(traj.nSamples());
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
  Re2       sdc = ss.reshape(Sz2{traj.nSamples(), 1}).broadcast(Sz2{1, traj.nTraces()});
  return sdc;
}

Re2 Radial3D(Trajectory const &traj, Index const lores)
{
  Log::Print("Calculating 3D radial analytic SDC");
  auto const nS = traj.nSamples();
  auto const nLo = std::abs(lores);
  auto const nHi = traj.nTraces() - nLo;

  Eigen::ArrayXf mergeHi = Eigen::ArrayXf::Ones(nS);
  Eigen::ArrayXf mergeLo;
  if (lores) {
    // From WASPI Paper (Wu et al 2007)
    float const lores_scale =
      (nHi / nLo) * Norm(traj.point(nS - 1, (lores > 0) ? nLo : 0)) / Norm(traj.point(nS - 1, (lores > 0) ? 0 : nHi));
    auto const  k1 = traj.point(0, (lores > 0) ? nLo : 0);
    auto const  k2 = traj.point(1, (lores > 0) ? nLo : 0);
    Index const gap = Norm(k1) / Norm(k2 - k1);
    mergeLo = Eigen::ArrayXf::LinSpaced(nS, 0, nS - 1) / lores_scale - (gap - 1);
    mergeLo = (mergeLo > 0).select(mergeLo, 0);
    mergeLo = (mergeLo < 1).select(mergeLo, 1);
    mergeLo = (1 - mergeLo) / lores_scale; // Match intensities of k-space
  }

  auto spoke_sdc = [&](Index const &spoke, Index const N, Eigen::ArrayXf const &merge) -> Re1 {
    // When k-space becomes undersampled need to flatten DC (Menon & Pipe 1999)
    auto const  mm = *std::max_element(traj.matrix().begin(), traj.matrix().end());
    float const R = (M_PI * mm * mm) / N; // Approx acceleration
    float const flat_start = nS / R;
    float const V = 1.f / (3. * (flat_start * flat_start) + 1. / 4.);
    Re1         sdc(nS);
    for (Index ir = 0; ir < nS; ir++) {
      float const rad = nS * Norm(traj.point(ir, spoke));
      if (rad == 0.f) {
        sdc(ir) = merge(ir) * V * 1.f / 8.f;
      } else if (rad < flat_start) {
        sdc(ir) = merge(ir) * V * (3.f * (rad * rad) + 1.f / 4.f);
      } else {
        sdc(ir) = merge(ir);
      }
    }
    return sdc;
  };

  Re2 sdc(nS, traj.nTraces());
  if (lores) {
    Re2 const sLo = spoke_sdc((lores > 0) ? 0 : nHi, nLo, mergeLo).reshape(Sz2{nS, 1}).broadcast(Sz2{1, nLo});
    Re2 const sHi = spoke_sdc((lores > 0) ? lores : 0, nHi, mergeHi).reshape(Sz2{nS, 1}).broadcast(Sz2{1, nHi});
    if (lores < 0) {
      sdc = sHi.concatenate(sLo, 1);
    } else {
      sdc = sLo.concatenate(sHi, 1);
    }
  } else {
    sdc = spoke_sdc(0, traj.nTraces(), mergeHi).reshape(Sz2{nS, 1}).broadcast(Sz2{1, traj.nTraces()});
  }
  return sdc;
}

auto Choose(SDC::Opts &opts, Index const nC, Trajectory const &traj, std::string const &ktype, float const os)
  -> std::shared_ptr<TensorOperator<Cx, 3>>
{
  Re2        sdc(traj.nSamples(), traj.nTraces());
  auto const iname = opts.type.Get();
  Sz3 const  dims{nC, traj.nSamples(), traj.nTraces()};
  if (iname == "" || iname == "none") {
    Log::Print("Using no density compensation");
    return nullptr;
  } else if (iname == "pipe") {
    if (traj.nDims() == 2) {
      sdc = SDC::Pipe<2>(traj, ktype, os, opts.maxIterations.Get(), opts.pow.Get());
    } else {
      sdc = SDC::Pipe<3>(traj, ktype, os, opts.maxIterations.Get(), opts.pow.Get());
    }
  } else {
    HD5::Reader reader(iname);
    sdc = reader.readTensor<Re2>(HD5::Keys::Weights);
    if (sdc.dimension(0) != traj.nSamples() || sdc.dimension(1) != traj.nTraces()) {
      Log::Fail("SDC dimensions on disk {}x{} did not match info {}x{}", sdc.dimension(0), sdc.dimension(1), traj.nSamples(),
                traj.nTraces());
    }
  }
  return std::make_shared<TensorScale<Cx, 3>>(dims, sdc.cast<Cx>());
}

} // namespace SDC
} // namespace rl
