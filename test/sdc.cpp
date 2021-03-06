#include "../src/gridder.h"
#include "../src/log.h"
#include "../src/sdc.h"
#include "../src/trajectory.h"
#include <catch2/catch.hpp>

TEST_CASE("SDC-Pipe", "[SDC]")
{
  Log log;
  Info info{.type = Info::Type::ThreeD,
            .channels = 1,
            .matrix = {4, 4, 4},
            .read_points = 4,
            .read_gap = 0,
            .spokes_hi = 1,
            .spokes_lo = 0,
            .lo_scale = 1};
  float const osamp = 2.f;
  R3 points(3, 4, 1);
  points.setZero();
  points(0, 0, 0) = 0.f;
  points(0, 1, 0) = 1.f / 3.f;
  points(0, 2, 0) = 2.f / 3.f;
  points(0, 3, 0) = 3.f / 3.f;
  Trajectory traj(info, points, log);

  SECTION("NN")
  {
    Kernel *kernel = new NearestNeighbour();
    Gridder gridder(traj, osamp, kernel, false, log);
    R2 sdc = SDC::Pipe(traj, gridder, log);
    CHECK(sdc.dimension(0) == info.read_points);
    CHECK(sdc.dimension(1) == info.spokes_total());
    Cx4 cart = gridder.newMultichannel(info.channels);
    CHECK(sdc(0, 0) == Approx(1.f));
    CHECK(sdc(1, 0) == Approx(1.f));
    CHECK(sdc(2, 0) == Approx(1.f));
    CHECK(sdc(3, 0) == Approx(1.f));
  }
}
