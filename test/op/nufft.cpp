#include "../../src/op/nufft.hpp"
#include "../../src/sdc.h"
#include "../../src/tensorOps.h"
#include "../../src/traj_spirals.h"
#include "../../src/trajectory.h"

#include <catch2/catch.hpp>

TEST_CASE("ops-nufft", "[ops]")
{
  Index const M = 16;
  float const os = 2.f;
  Info const info{
    .type = Info::Type::ThreeD,
    .matrix = Eigen::Array3l::Constant(M),
    .channels = 1,
    .read_points = Index(os * M / 2),
    .spokes = Index(M * M),
    .volumes = 1,
    .echoes = 1,
    .tr = 1.f,
    .voxel_size = Eigen::Array3f::Constant(1.f),
    .origin = Eigen::Array3f::Constant(0.f),
    .direction = Eigen::Matrix3f::Identity()};
  auto const points = ArchimedeanSpiral(info.read_points, info.spokes);
  Trajectory const traj(info, points);
  auto const nn = make_kernel("NN", info.type, os);
  auto const m1 = traj.mapping(1, os);
  auto grid = make_grid(nn.get(), m1, false);
  grid->setSDC(SDC::Pipe(traj, true, os));

  auto nufft = NUFFTOp(Sz3{M, M, M}, grid.get());
  auto const dims = nufft.inputDimensions();
  Cx5 x(dims), y(dims);
  Cx3 r(info.channels, info.read_points, info.spokes);

  /* See note in grid.cpp - classic Dot test isn't appropriate.
   * Give this test a decent amount of tolerance because the cropping/padding means we don't
   * get back exactly what we should
   */
  SECTION("SDC-Full")
  {
    grid->setSDCPower(1.0f);
    x.setRandom();
    y = nufft.AdjA(x);
    auto const xy = Dot(x, y);
    auto const yy = Dot(y, y);
    CHECK(std::abs((yy - xy) / (yy + xy + 1.e-15f)) == Approx(0).margin(1.e-1));
  }

  /*
   * These two are waaaaaaay out because of how bad the condition number of the system is
   * Keep them in for now as the difference appears stable
   */
  SECTION("SDC-Half")
  {
    grid->setSDCPower(0.5f);
    x.setRandom();
    y = nufft.AdjA(x);
    auto const xy = Dot(x, y);
    auto const yy = Dot(y, y);
    CHECK(std::abs((yy - xy) / (yy + xy + 1.e-15f)) == Approx(0.86f).margin(1.e-2));
  }

  SECTION("SDC-None")
  {
    grid->setSDCPower(0.0f);
    x.setRandom();
    y = nufft.AdjA(x);
    auto const xy = Dot(x, y);
    auto const yy = Dot(y, y);
    CHECK(std::abs((yy - xy) / (yy + xy + 1.e-15f)) == Approx(0.99f).margin(1.e-2));
  }
}