#include "../src/sdc.h"
#include "../src/log.h"
#include "../src/op/grid-kernel.hpp"
#include "../src/trajectory.h"
#include <catch2/catch.hpp>

TEST_CASE("NN", "[SDC]")
{
  Log log;
  Info info{
    .type = Info::Type::ThreeDStack,
    .channels = 1,
    .matrix = {4, 4, 1},
    .read_points = 4,
    .read_gap = 0,
    .spokes_hi = 1,
    .spokes_lo = 0,
    .lo_scale = 1};
  R3 points(3, 4, 1);
  points.setZero();
  points(0, 0, 0) = 0.5f * 0.f;
  points(0, 1, 0) = 0.5f * 1.f / 3.f;
  points(0, 2, 0) = 0.5f * 2.f / 3.f;
  points(0, 3, 0) = 0.5f * 3.f / 3.f;
  Trajectory traj(info, points, log);

  SECTION("Pipe")
  {
    R2 sdc = SDC::Pipe(traj, log);
    CHECK(sdc.dimension(0) == info.read_points);
    CHECK(sdc.dimension(1) == info.spokes_total());
    CHECK(sdc(0, 0) == Approx(1.04336f).margin(1.e-4f));
    CHECK(sdc(1, 0) == Approx(1.04336f).margin(1.e-4f));
    // Next two points excluded by margin at edge of grid
    CHECK(sdc(2, 0) == Approx(0.0f).margin(1.e-4f));
    CHECK(sdc(3, 0) == Approx(0.0f).margin(1.e-4f));
  }
}
