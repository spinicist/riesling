#include "sdc.h"

#include "io.h"
#include "kernel.hpp"
#include "op/grid-echo.hpp"
#include "tensorOps.h"
#include "threads.h"
#include "trajectory.h"

namespace SDC {

R2 Pipe(Trajectory const &traj, bool const nn, float const os)
{
  Log::Print("Using Pipe/Zwart/Menon SDC...");
  auto const info = traj.info();
  Cx3 W(1, info.read_points, info.spokes);
  Cx3 Wp(W.dimensions());

  std::unique_ptr<Kernel> k; // Need to keep this alive until the end of the function
  std::unique_ptr<GridBase> gridder;
  if (nn) {
    k = std::make_unique<NearestNeighbour>();
    auto const m = traj.mapping(1, os);
    gridder =
      std::make_unique<GridEcho<1, 1>>(dynamic_cast<SizedKernel<1, 1> const *>(k.get()), m, false);
  } else {
    auto const m = traj.mapping(3, os);
    if (info.type == Info::Type::ThreeD) {
      k = std::make_unique<PipeSDC<5, 5>>(os);
      gridder = std::make_unique<GridEcho<5, 5>>(
        dynamic_cast<SizedKernel<5, 5> const *>(k.get()), m, false);
    } else {
      k = std::make_unique<PipeSDC<5, 1>>(os);
      gridder = std::make_unique<GridEcho<5, 1>>(
        dynamic_cast<SizedKernel<5, 1> const *>(k.get()), m, false);
    }
  }
  gridder->doNotWeightEchoes();
  W.setConstant(1.f);

  Cx5 temp(gridder->inputDimensions(1));
  for (Index ii = 0; ii < 40; ii++) {
    Wp.setZero();
    temp.setZero();
    gridder->Adj(W, temp);
    gridder->A(temp, Wp);
    Wp.device(Threads::GlobalDevice()) =
      (Wp.real() > 0.f).select(W / Wp, Wp.constant(0.f)).eval(); // Avoid divide by zero problems
    float const delta = R0((Wp - W).real().abs().maximum())();
    W.device(Threads::GlobalDevice()) = Wp;
    if (delta < 1e-6) {
      Log::Print("SDC converged, delta was {}", delta);
      break;
    } else {
      Log::Print("SDC Delta {}", delta);
    }
  }
  if (!nn && (info.read_points > 6)) {
    // At this point we have relative weights. There might be something odd going on at the end of
    // the spokes. Count back from the ends to miss that and then average.
    W = W / W.constant(Mean(W.slice(Sz3{0, info.read_points - 6, 0}, Sz3{1, 1, info.spokes})));
  }
  Log::Print("SDC finished.");
  return W.real().chip<0>(0);
}

R2 Radial2D(Trajectory const &traj)
{
  Log::Print(FMT_STRING("Calculating 2D radial analytic SDC"));
  Info const &info = traj.info();
  auto spoke_sdc = [&](Index const spoke, Index const N) -> R1 {
    float const k_delta = (traj.point(1, spoke, 1.f) - traj.point(0, spoke, 1.f)).norm();
    float const V = 2.f * k_delta * M_PI / N; // Area element
    // When k-space becomes undersampled need to flatten DC (Menon & Pipe 1999)
    float const R = (M_PI * info.matrix.maxCoeff()) / N;
    float const flat_start = info.read_points / sqrt(R);
    float const flat_val = V * flat_start;
    R1 sdc(info.read_points);
    for (Index ir = 0; ir < info.read_points; ir++) {
      float const rad = traj.point(ir, spoke, info.read_points).norm();
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

  R1 const ss = spoke_sdc(0, info.spokes);
  R2 sdc = ss.reshape(Sz2{info.read_points, 1}).broadcast(Sz2{1, info.spokes});
  return sdc;
}

R2 Radial3D(Trajectory const &traj, Index const lores, Index const gap)
{
  Log::Print(FMT_STRING("Calculating 2D radial analytic SDC"));
  auto const &info = traj.info();

  Eigen::ArrayXf ind = Eigen::ArrayXf::LinSpaced(info.read_points, 0, info.read_points - 1);
  Eigen::ArrayXf mergeHi = ind - (gap - 1);
  mergeHi = (mergeHi > 0).select(mergeHi, 0);
  mergeHi = (mergeHi < 1).select(mergeHi, 1);

  Eigen::ArrayXf mergeLo;
  if (lores) {
    float const scale = traj.point(info.read_points - 1, lores, 1.f).norm() /
                        traj.point(info.read_points - 1, 0, 1.f).norm();
    mergeLo = ind / scale - (gap - 1);
    mergeLo = (mergeLo > 0).select(mergeLo, 0);
    mergeLo = (mergeLo < 1).select(mergeLo, 1);
    mergeLo = (1 - mergeLo) / scale; // Match intensities of k-space
    mergeLo.head(gap) = 0.;          // Don't touch these points
  }

  auto spoke_sdc = [&](Index const &spoke, Index const N) -> R1 {
    // Calculate the point spacing
    float const k_delta = (traj.point(1, spoke, 1.f) - traj.point(0, spoke, 1.f)).norm();
    float const V = (4.f / 3.f) * k_delta * M_PI / N; // Volume element
    // When k-space becomes undersampled need to flatten DC (Menon & Pipe 1999)
    float const R = (M_PI * info.matrix.maxCoeff() * info.matrix.maxCoeff()) / N;
    float const flat_start = info.read_points / sqrt(R);
    float const flat_val = V * (3. * (flat_start * flat_start) + 1. / 4.);
    R1 sdc(info.read_points);
    for (Index ir = 0; ir < info.read_points; ir++) {
      float const rad = traj.point(ir, spoke, info.read_points).norm();
      float const merge = (spoke < lores) ? mergeLo(ir) : mergeHi(ir);
      if (rad == 0.f) {
        sdc(ir) = merge * V * 1.f / 8.f;
      } else if (rad < flat_start) {
        sdc(ir) = merge * V * (3.f * (rad * rad) + 1.f / 4.f);
      } else {
        sdc(ir) = merge * flat_val;
      }
    }
    return sdc;
  };

  R2 sdc(info.read_points, info.spokes);
  if (lores) {
    R1 const ss = spoke_sdc(0, lores);
    sdc.slice(Sz2{0, 0}, Sz2{info.read_points, lores}) =
      ss.reshape(Sz2{info.read_points, 1}).broadcast(Sz2{1, lores});
  }
  R1 const ss = spoke_sdc(lores, info.spokes - lores);
  sdc.slice(Sz2{0, lores}, Sz2{info.read_points, info.spokes - lores}) =
    ss.reshape(Sz2{info.read_points, 1}).broadcast(Sz2{1, info.spokes - lores});

  return sdc;
}

R2 Radial(Trajectory const &traj, Index const lores, Index const gap)
{

  if (traj.info().type == Info::Type::ThreeD) {
    return Radial3D(traj, lores, gap);
  } else {
    return Radial2D(traj);
  }
}

R2 Choose(std::string const &iname, Trajectory const &traj, float const os)
{
  R2 sdc(traj.info().read_points, traj.info().spokes);
  if (iname == "") {
    Log::Print("Using no density compensation");
    sdc.setConstant(1.f);
  } else if (iname == "none") {
    Log::Print("Using no density compensation");
    sdc.setConstant(1.f);
  } else if (iname == "pipe") {
    sdc = Pipe(traj, false, 2.1f);
  } else if (iname == "pipenn") {
    sdc = Pipe(traj, true, os);
  } else {
    HD5::Reader reader(iname);
    auto const sdcInfo = reader.readInfo();
    auto const trajInfo = traj.info();
    if (sdcInfo.read_points != trajInfo.read_points || sdcInfo.spokes != trajInfo.spokes) {
      Log::Fail(
        "SDC trajectory dimensions {}x{} do not match main trajectory {}x{}",
        sdcInfo.read_points,
        sdcInfo.spokes,
        trajInfo.read_points,
        trajInfo.spokes);
    }
    sdc = reader.readSDC(sdcInfo);
  }
  return sdc;
}

} // namespace SDC
