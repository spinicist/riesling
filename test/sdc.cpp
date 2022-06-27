#include "../src/sdc.h"
#include "../src/log.h"
#include "../src/traj_spirals.h"
#include "../src/trajectory.h"
#include <catch2/catch.hpp>

TEST_CASE("SDC")
{
  Index const M = 32;
  float const os = 2.f;
  Info const info{
    .type = Info::Type::ThreeD,
    .matrix = Eigen::Array3l::Constant(M),
    .channels = 1,
    .read_points = Index(os * M / 2),
    .spokes = Index(M * M / 4),
    .volumes = 1,
    .frames = 1,
    .tr = 1.f,
    .voxel_size = Eigen::Array3f::Constant(1.f),
    .origin = Eigen::Array3f::Constant(0.f),
    .direction = Eigen::Matrix3f::Identity()};
  auto const points = ArchimedeanSpiral(info.read_points, info.spokes);
  Trajectory const traj(info, points);

  SECTION("Pipe-NN")
  {
    R2 sdc = SDC::Pipe(traj, true, 2.1f);
    CHECK(sdc.dimension(0) == info.read_points);
    CHECK(sdc.dimension(1) == info.spokes);
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
    R2 sdc = SDC::Pipe(traj, false, 2.1f);
    CHECK(sdc.dimension(0) == info.read_points);
    CHECK(sdc.dimension(1) == info.spokes);
    // Central points should be very small
    CHECK(sdc(0, 0) == Approx(0.00106f).margin(1.e-4f));
    CHECK(sdc(1, 0) == Approx(0.00427f).margin(1.e-4f));
    // Undersampled points should be close to one
    CHECK(sdc(25, 0) == Approx(0.81739f).margin(1.e-1f));
    CHECK(sdc(26, 0) == Approx(0.8954f).margin(1.e-1f));
    CHECK(sdc(31, 0) == Approx(1.77723f).margin(1.e-4f));
  }

}
