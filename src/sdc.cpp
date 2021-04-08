#include "sdc.h"

#include "gridder.h"
#include "kernel.h"
#include "tensorOps.h"

std::unordered_map<std::string, SDC> SDCMap{
    {"none", SDC::None}, {"ana", SDC::Analytic}, {"pipe", SDC::Pipe}};

Cx2 SDCPipe(Info const &info, Gridder *gridder, Kernel *kernel, Log &log)
{
  log.info("Using Pipe/Zwart/Menon SDC...");
  Cx2 W(info.read_points, info.spokes_total());
  Cx2 Wp(info.read_points, info.spokes_total());

  W.setConstant(1.f);
  kernel->sqrtOn();
  Cx3 temp = gridder->newGrid1();
  for (long ii = 0; ii < 8; ii++) {
    Wp.setZero();
    temp.setZero();
    gridder->toCartesian(W, temp);
    gridder->toNoncartesian(temp, Wp);
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
  return W;
}