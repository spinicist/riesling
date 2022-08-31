#include "../src/sdc.h"
#include "../src/log.h"
#include "../src/traj_spirals.h"
#include "../src/trajectory.h"
#include <catch2/catch.hpp>

using namespace rl;

TEST_CASE("SDC", "[sdc]")
{
  Log::SetLevel(Log::Level::Testing);
  Index const M = 16;
  Info const info{.channels = 1, .samples = 3, .traces = 1, .matrix = Sz3{M, M, M}};
  Re3 points(3, 3, 1);
  points.setZero();
  points(0, 1, 0) = -0.25;
  points(0, 2, 0) =  0.25;
  Trajectory const traj(info, points);

  SECTION("Pipe-NN")
  {
    Re2 sdc = SDC::Pipe(traj, true, 2.f);
    CHECK(sdc.dimension(0) == info.samples);
    CHECK(sdc.dimension(1) == info.traces);
    CHECK(sdc(0, 0) == Approx(1.f).margin(1.e-6f));
    CHECK(sdc(1, 0) == Approx(1.f).margin(1.e-6f));
    CHECK(sdc(2, 0) == Approx(1.f).margin(1.e-6f));
  }

  SECTION("Pipe")
  {
    Re2 sdc = SDC::Pipe(traj, false, 2.1f);
    CHECK(sdc.dimension(0) == info.samples);
    CHECK(sdc.dimension(1) == info.traces);
    CHECK(sdc(0, 0) == Approx(1.f).margin(1.e-6f));
    CHECK(sdc(1, 0) == Approx(1.f).margin(1.e-6f));
    CHECK(sdc(2, 0) == Approx(1.f).margin(1.e-6f));
  }
}
