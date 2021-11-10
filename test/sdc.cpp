#include "../src/sdc.h"
#include "../src/log.h"
#include "../src/op/grid-kb.h"
#include "../src/op/grid-nn.h"
#include "../src/trajectory.h"
#include <catch2/catch.hpp>

TEST_CASE("Pipe-NN", "[SDC]")
{
  Log log;
  Info info{
    .type = Info::Type::ThreeD,
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
  points(0, 0, 0) = 0.5f * 0.f;
  points(0, 1, 0) = 0.5f * 1.f / 3.f;
  points(0, 2, 0) = 0.5f * 2.f / 3.f;
  points(0, 3, 0) = 0.5f * 3.f / 3.f;
  Trajectory traj(info, points, log);

  SECTION("NN")
  {
    std::unique_ptr<GridOp> gridder = std::make_unique<GridNN>(traj, osamp, false, log);
    R2 sdc = SDC::Pipe(traj, gridder, log);
    CHECK(sdc.dimension(0) == info.read_points);
    CHECK(sdc.dimension(1) == info.spokes_total());
    CHECK(sdc(0, 0) == Approx(1.f));
    CHECK(sdc(1, 0) == Approx(1.f));
    CHECK(sdc(2, 0) == Approx(1.f));
    CHECK(sdc(3, 0) == Approx(1.f));
  }
}

TEST_CASE("Pipe-KB", "[SDC]")
{
  Log log;
  Info info{
    .type = Info::Type::ThreeD,
    .channels = 1,
    .matrix = {16, 16, 16},
    .read_points = 4,
    .read_gap = 0,
    .spokes_hi = 1,
    .spokes_lo = 0,
    .lo_scale = 1};
  float const osamp = 3.f;
  R3 points(3, 4, 1);
  points.setZero();
  points(0, 0, 0) = 0.5f * 0.f;
  points(0, 1, 0) = 0.5f * 1.f / 16.f;
  points(0, 2, 0) = 0.5f * 2.f / 16.f;
  points(0, 3, 0) = 0.5f * 8.f / 16.f;
  Trajectory traj(info, points, log);
  std::unique_ptr<GridOp> gridder = std::make_unique<GridKB<5, 5>>(traj, osamp, false, log);

  SECTION("KB")
  {

    gridder->sqrtOn();
    auto nc = info.noncartesianVolume();
    auto c = gridder->newMultichannel(1);
    nc.setConstant(1.f);
    c.setZero();
    gridder->Adj(nc, c);
    gridder->A(c, nc);
    R3 realPart = nc.real();
    CHECK(realPart(0, 0, 0) == Approx(0.24551f).margin(1.e-3f));
    CHECK(realPart(0, 3, 0) == Approx(0.19692f).margin(1.e-4f));
    gridder->sqrtOff();
  }
  SECTION("KB-SDC")
  {
    R2 sdc = SDC::Pipe(traj, gridder, log);
    CHECK(sdc.dimension(0) == info.read_points);
    CHECK(sdc.dimension(1) == info.spokes_total());
    CHECK(sdc(0, 0) == Approx(4.07837f).margin(1.e-4f));
    CHECK(sdc(1, 0) == Approx(4.05309f).margin(1.e-4f));
    CHECK(sdc(2, 0) == Approx(4.86068f).margin(1.e-4f));
    CHECK(sdc(3, 0) == Approx(5.07831f).margin(1.e-4f));
  }
}
