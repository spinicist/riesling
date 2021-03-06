#include "sdc.h"

#include "gridder.h"
#include "io_hd5.h"
#include "kernel.h"
#include "tensorOps.h"
#include "trajectory.h"

namespace SDC {
void Load(std::string const &iname, Trajectory const &traj, Gridder &gridder, Log &log)
{
  if (iname == "") {
    return;
  } else if (iname == "none") {
    return;
  } else if (iname == "pipe") {
    gridder.setSDC(Pipe(traj, gridder, log));
  } else if (iname == "radial") {
    gridder.setSDC(Radial(traj, log));
  } else {
    HD5::Reader reader(iname, log);
    gridder.setSDC(reader.readSDC(reader.readInfo()));
  }
}

R2 Pipe(Trajectory const &traj, Gridder &gridder, Log &log)
{
  log.info("Using Pipe/Zwart/Menon SDC...");
  Cx3 W(1, gridder.info().read_points, gridder.info().spokes_total());
  Cx3 Wp(W.dimensions());

  W.setZero();
  for (long is = 0; is < traj.info().spokes_total(); is++) {
    for (long ir = 0; ir < traj.info().read_points; ir++) {
      W(0, ir, is) = traj.merge(ir, is);
    }
  }

  gridder.kernel()->sqrtOn();
  Cx4 temp = gridder.newMultichannel(1);
  for (long ii = 0; ii < 10; ii++) {
    Wp.setZero();
    temp.setZero();
    gridder.toCartesian(W, temp);
    gridder.toNoncartesian(temp, Wp);
    Wp.device(Threads::GlobalDevice()) =
        (Wp.real() > 0.f).select(W / Wp, W); // Avoid divide by zero problems
    float const delta = R0((Wp - W).real().square().maximum())();
    W.device(Threads::GlobalDevice()) = Wp;
    if (delta < 5.e-2) {
      log.info("SDC converged, delta was {}", delta);
      break;
    } else {
      log.info("SDC Delta {}", delta);
    }
  }
  gridder.kernel()->sqrtOff();
  log.info("SDC finished.");
  return W.real().chip(0, 0);
}

R2 Radial2D(Trajectory const &traj, Log &log)
{
  Info const &info = traj.info();
  auto spoke_sdc = [&](long const spoke, long const N, float const scale) -> R1 {
    float const k_delta = 1. / (info.read_oversamp() * scale);
    float const V = 2.f * k_delta * M_PI / N; // Area element
    // When k-space becomes undersampled need to flatten DC (Menon & Pipe 1999)
    float const R = (M_PI * info.matrix.maxCoeff()) / (N * scale);
    float const flat_start = info.read_points / sqrt(R);
    float const flat_val = V * flat_start;
    R1 sdc(info.read_points);
    for (long ir = 0; ir < info.read_points; ir++) {
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

  R2 sdc(info.read_points, info.spokes_total());
  if (info.spokes_lo) {
    R1 const ss = spoke_sdc(0, info.spokes_lo, info.lo_scale);
    sdc.slice(Sz2{0, 0}, Sz2{info.read_points, info.spokes_lo}) =
        ss.reshape(Sz2{info.read_points, 1}).broadcast(Sz2{1, info.spokes_lo});
  }
  R1 const ss = spoke_sdc(info.spokes_lo, info.spokes_hi, 1.f);
  sdc.slice(Sz2{0, info.spokes_lo}, Sz2{info.read_points, info.spokes_hi}) =
      ss.reshape(Sz2{info.read_points, 1}).broadcast(Sz2{1, info.spokes_hi});

  log.info(FMT_STRING("Calculated 2D radial analytic SDC"));
  return sdc;
}

R2 Radial3D(Trajectory const &traj, Log &log)
{
  auto const &info = traj.info();
  auto spoke_sdc = [&](long const &spoke, long const N, float const scale) -> R1 {
    // Calculate the point spacing
    float const k_delta = 1.f / (info.read_oversamp() * scale);
    float const V = (4.f / 3.f) * k_delta * M_PI / N; // Volume element
    // When k-space becomes undersampled need to flatten DC (Menon & Pipe 1999)
    float const R = (M_PI * info.matrix.maxCoeff() * info.matrix.maxCoeff()) / (N * scale * scale);
    float const flat_start = info.read_points / sqrt(R);
    float const flat_val = V * (3. * (flat_start * flat_start) + 1. / 4.);
    R1 sdc(info.read_points);
    for (long ir = 0; ir < info.read_points; ir++) {
      float const rad = traj.point(ir, spoke, info.read_points).norm();
      float const merge = traj.merge(ir, spoke);
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

  R2 sdc(info.read_points, info.spokes_total());
  if (info.spokes_lo) {
    R1 const ss = spoke_sdc(0, info.spokes_lo, info.lo_scale);
    sdc.slice(Sz2{0, 0}, Sz2{info.read_points, info.spokes_lo}) =
        ss.reshape(Sz2{info.read_points, 1}).broadcast(Sz2{1, info.spokes_lo});
  }
  R1 const ss = spoke_sdc(info.spokes_lo, info.spokes_hi, 1.f);
  sdc.slice(Sz2{0, info.spokes_lo}, Sz2{info.read_points, info.spokes_hi}) =
      ss.reshape(Sz2{info.read_points, 1}).broadcast(Sz2{1, info.spokes_hi});

  log.info(FMT_STRING("Calculated 2D radial analytic SDC"));
  return sdc;
}

R2 Radial(Trajectory const &traj, Log &log)
{

  if (traj.info().type == Info::Type::ThreeD) {
    return Radial3D(traj, log);
  } else {
    return Radial2D(traj, log);
  }
}

} // namespace SDC
