#include "../src/sdc.h"
#include "../src/log.h"
#include "../src/op/grid-kernel.hpp"
#include "../src/traj_spirals.h"
#include "../src/trajectory.h"
#include <catch2/catch.hpp>

TEST_CASE("NN", "[SDC]")
{
  Log log;
  long const M = 32;
  float const os = 2.f;
  Info const info{
    .type = Info::Type::ThreeD,
    .channels = 1,
    .matrix = Eigen::Array3l::Constant(M),
    .read_points = long(os * M / 2),
    .read_gap = 0,
    .spokes_hi = long(M * M / 4),
    .spokes_lo = 0,
    .lo_scale = 1.f,
    .volumes = 1,
    .echoes = 1,
    .tr = 1.f,
    .voxel_size = Eigen::Array3f::Constant(1.f),
    .origin = Eigen::Array3f::Constant(0.f),
    .direction = Eigen::Matrix3f::Identity()};
  auto const points = ArchimedeanSpiral(info);
  Trajectory const traj(info, points, log);

  SECTION("Pipe")
  {
    R2 sdc = SDC::Pipe(traj, false, 2.1f, log);
    CHECK(sdc.dimension(0) == info.read_points);
    CHECK(sdc.dimension(1) == info.spokes_total());
    // Central points should be very small
    CHECK(sdc(0, 0) == Approx(0.00129f).margin(1.e-4f));
    CHECK(sdc(1, 0) == Approx(0.00519f).margin(1.e-4f));
    // Undersampled points should be one
    CHECK(sdc(25, 0) == Approx(1.0f).margin(1.e-1f));
    CHECK(sdc(26, 0) == Approx(1.0f).margin(1.e-1f));
    // Point excluded by margin at edge of grid
    CHECK(sdc(31, 0) == Approx(0.0f).margin(1.e-4f));
  }
}
