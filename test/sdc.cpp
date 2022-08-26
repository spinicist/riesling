#include "../src/sdc.h"
#include "../src/log.h"
#include "../src/traj_spirals.h"
#include "../src/trajectory.h"
#include <catch2/catch.hpp>

using namespace rl;

TEST_CASE("SDC", "[sdc]")
{
  Index const M = 32;
  float const os = 2.f;
  Info const info{.channels = 1, .samples = Index(os * M / 2), .traces = Index(M * M), .matrix = Sz3{M, M, M}};
  auto const points = ArchimedeanSpiral(info.samples, info.traces);
  Trajectory const traj(info, points);

  SECTION("Pipe-NN")
  {
    Re2 sdc = SDC::Pipe(traj, true, 2.1f);
    CHECK(sdc.dimension(0) == info.samples);
    CHECK(sdc.dimension(1) == info.traces);
    // Central points should be very small
    CHECK(sdc(0, 0) == Approx(0.00042f).margin(1.e-4f));
    CHECK(sdc(1, 0) == Approx(0.00568f).margin(1.e-4f));
    // Undersampled points should be constant
    CHECK(sdc(25, 0) == Approx(0.10798f).margin(1.e-1f));
    CHECK(sdc(26, 0) == Approx(0.10798f).margin(1.e-1f));
    CHECK(sdc(31, 0) == Approx(0.10798f).margin(1.e-4f));
  }

  SECTION("Pipe")
  {
    Re2 sdc = SDC::Pipe(traj, false, 2.1f);
    CHECK(sdc.dimension(0) == info.samples);
    CHECK(sdc.dimension(1) == info.traces);
    // Central points should be very small
    CHECK(sdc(0, 0) == Approx(0.00122f).margin(1.e-4f));
    CHECK(sdc(1, 0) == Approx(0.00472f).margin(1.e-4f));
    // Undersampled points should be close to one
    CHECK(sdc(25, 0) == Approx(0.94893f).margin(1.e-1f));
    CHECK(sdc(26, 0) == Approx(0.8954f).margin(1.e-1f));
    CHECK(sdc(31, 0) == Approx(2.20694f).margin(1.e-4f));
  }
}
