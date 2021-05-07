#include "sdc.h"

#include "gridder.h"
#include "io_hd5.h"
#include "kernel.h"
#include "tensorOps.h"

namespace SDC {
void Load(
    std::string const &fname,
    Info const &info,
    R3 const &traj,
    Kernel *kernel,
    Gridder &gridder,
    Log &log)
{
  if (fname == "") {
    return;
  } else if (fname == "none") {
    return;
  } else if (fname == "pipe") {
    R2 const sdc = Pipe(info, gridder, kernel, log);
    gridder.setSDC(sdc);
  } else if (fname == "radial") {
    R2 const sdc = Radial(info, traj, log);
    gridder.setSDC(sdc);
  } else {
    HD5::Reader reader(fname, log);
    gridder.setSDC(reader.readSDC());
  }
}

R2 Pipe(Info const &info, Gridder &gridder, Kernel *kernel, Log &log)
{
  log.info("Using Pipe/Zwart/Menon SDC...");
  Cx2 W(info.read_points, info.spokes_total());
  Cx2 Wp(info.read_points, info.spokes_total());

  W.setConstant(1.f);
  kernel->sqrtOn();
  Cx3 temp = gridder.newGrid1();
  for (long ii = 0; ii < 8; ii++) {
    Wp.setZero();
    temp.setZero();
    gridder.toCartesian(W, temp);
    gridder.toNoncartesian(temp, Wp);
    Wp.device(Threads::GlobalDevice()) =
        (Wp.real() > 0.f).select(W / Wp, W); // Avoid divide by zero problems
    float const delta = Norm(Wp - W) / W.size();
    W.device(Threads::GlobalDevice()) = Wp;
    if (delta < 1.e-4) {
      log.info("SDC converged, delta was {}", delta);
      break;
    } else {
      log.info("SDC Delta {}", delta);
    }
  }
  kernel->sqrtOff();
  log.info("SDC finished.");
  return W.real();
}

// Helper function to convert Tensor to Point
inline Point3 toCart(R1 const &p, float const xyScale, float const zScale)
{
  return Point3{p(0) * xyScale, p(1) * xyScale, p(2) * zScale};
}

R2 Radial2D(Info const &info, R3 const &traj, Log &log)
{
  // Calculate the point spacing
  float const spokeOversamp = info.read_points / (info.matrix.maxCoeff() / 2);

  auto spoke_sdc = [&](R2 const &spoke, long const N, float const scale) -> R1 {
    float const k_delta = 1. / (spokeOversamp * scale);
    float const V = 2.f * k_delta * M_PI / N; // Area element
    // When k-space becomes undersampled need to flatten DC (Menon & Pipe 1999)
    float const R = (M_PI * info.matrix.maxCoeff()) / (N * scale);
    float const flat_start = info.read_points / sqrt(R);
    float const flat_val = V * flat_start;
    R1 sdc(info.read_points);
    for (long ir = 0; ir < info.read_points; ir++) {
      float const rad = toCart(spoke.chip(ir, 1), info.read_points, 1.f).norm();
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
    R1 const ss = spoke_sdc(traj.chip(0, 2), info.spokes_lo, info.lo_scale);
    sdc.slice(Sz2{0, 0}, Sz2{info.read_points, info.spokes_lo}) =
        ss.reshape(Sz2{info.read_points, 1}).broadcast(Sz2{1, info.spokes_lo});
  }
  R1 const ss = spoke_sdc(traj.chip(info.spokes_lo, 2), info.spokes_hi, 1.f);
  sdc.slice(Sz2{0, info.spokes_lo}, Sz2{info.read_points, info.spokes_hi}) =
      ss.reshape(Sz2{info.read_points, 1}).broadcast(Sz2{1, info.spokes_hi});

  log.info(FMT_STRING("Calculated 2D radial analytic SDC"));
  return sdc;
}

R2 Radial3D(Info const &info, R3 const &traj, Log &log)
{
  // Calculate the point spacing
  float const spokeOversamp = info.read_points / (info.matrix.maxCoeff() / 2);

  auto spoke_sdc = [&](R2 const &spoke, long const N, float const scale) -> R1 {
    // Calculate the point spacing
    float const k_delta = 1.f / (spokeOversamp * scale);
    float const V = (4.f / 3.f) * k_delta * M_PI / N; // Volume element
    // When k-space becomes undersampled need to flatten DC (Menon & Pipe 1999)
    float const R = (M_PI * info.matrix.maxCoeff() * info.matrix.maxCoeff()) / (N * scale * scale);
    float const flat_start = info.read_points / sqrt(R);
    float const flat_val = V * (3. * (flat_start * flat_start) + 1. / 4.);
    R1 sdc(info.read_points);
    for (long ir = 0; ir < info.read_points; ir++) {
      float const rad = toCart(spoke.chip(ir, 1), info.read_points, 1.f).norm();
      if (rad == 0.f) {
        sdc(ir) = V * 1.f / 8.f;
      } else if (rad < flat_start) {
        sdc(ir) = V * (3.f * (rad * rad) + 1.f / 4.f);
      } else {
        sdc(ir) = flat_val;
      }
    }
    return sdc;
  };

  R2 sdc(info.read_points, info.spokes_total());
  if (info.spokes_lo) {
    R1 const ss = spoke_sdc(traj.chip(0, 2), info.spokes_lo, info.lo_scale);
    sdc.slice(Sz2{0, 0}, Sz2{info.read_points, info.spokes_lo}) =
        ss.reshape(Sz2{info.read_points, 1}).broadcast(Sz2{1, info.spokes_lo});
  }
  R1 const ss = spoke_sdc(traj.chip(info.spokes_lo, 2), info.spokes_hi, 1.f);
  sdc.slice(Sz2{0, info.spokes_lo}, Sz2{info.read_points, info.spokes_hi}) =
      ss.reshape(Sz2{info.read_points, 1}).broadcast(Sz2{1, info.spokes_hi});

  log.info(FMT_STRING("Calculated 2D radial analytic SDC"));
  return sdc;
}

R2 Radial(Info const &info, R3 const &traj, Log &log)
{

  if (info.type == Info::Type::ThreeD) {
    return Radial3D(info, traj, log);
  } else {
    return Radial2D(info, traj, log);
  }
}

} // namespace SDC
