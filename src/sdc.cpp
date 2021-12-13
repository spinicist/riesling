#include "sdc.h"

#include "io.h"
#include "op/grid-kernel.hpp"
#include "op/grid-nn.h"
#include "tensorOps.h"
#include "threads.h"
#include "trajectory.h"

namespace SDC {

R2 Pipe(Trajectory const &traj, bool const nn, float const os, Log &log)
{
  log.info("Using Pipe/Zwart/Menon SDC...");
  auto const info = traj.info();
  Cx3 W(1, info.read_points, info.spokes);
  Cx3 Wp(W.dimensions());

  std::unique_ptr<GridOp> gridder;
  if (nn) {
    gridder = std::make_unique<GridNN>(traj, os, false, log);
  } else {
    if (info.type == Info::Type::ThreeD) {
      gridder = std::make_unique<Grid<PipeSDC<5, 5>>>(traj, os, false, log);
    } else {
      gridder = std::make_unique<Grid<PipeSDC<5, 1>>>(traj, os, false, log);
    }
  }
  W.setConstant(1.f);

  Cx5 temp(gridder->inputDimensions(1, info.echoes));
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
      log.info("SDC converged, delta was {}", delta);
      break;
    } else {
      log.info("SDC Delta {}", delta);
    }
  }
  if (!nn && (info.read_points > 6)) {
    // At this point we have relative weights. There might be something odd going on at the end of
    // the spokes. Count back from the ends to miss that and then average.
    W = W / W.constant(Mean(W.slice(Sz3{0, info.read_points - 6, 0}, Sz3{1, 1, info.spokes})));
  }
  log.info("SDC finished.");
  return W.real().chip<0>(0);
}

R2 Choose(std::string const &iname, Trajectory const &traj, float const os, Log &log)
{
  R2 sdc(traj.info().read_points, traj.info().spokes);
  if (iname == "") {
    log.info("Using no density compensation");
    sdc.setConstant(1.f);
  } else if (iname == "none") {
    log.info("Using no density compensation");
    sdc.setConstant(1.f);
  } else if (iname == "pipe") {
    sdc = Pipe(traj, false, 2.1f, log);
  } else if (iname == "pipenn") {
    sdc = Pipe(traj, true, os, log);
  } else {
    HD5::Reader reader(iname, log);
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
