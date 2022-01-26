#define CATCH_CONFIG_ENABLE_BENCHMARKING

#include "../src/op/grid.h"
#include "../src/info.h"
#include "../src/traj_spirals.h"

#include <catch2/catch.hpp>

Index const M = 32;
Index const C = 8;
Info const info{
  .type = Info::Type::ThreeD,
  .matrix = Eigen::Array3l::Constant(M),
  .channels = C,
  .read_points = M / 2,
  .spokes = M * M,
  .volumes = 1,
  .echoes = 1,
  .tr = 1.f,
  .voxel_size = Eigen::Array3f::Constant(1.f),
  .origin = Eigen::Array3f::Constant(0.f),
  .direction = Eigen::Matrix3f::Identity()};

auto nc = info.noncartesianVolume();

auto const points = ArchimedeanSpiral(info.read_points, info.spokes);
Trajectory traj(info, points);
float const os = 2.f;
auto const m1 = traj.mapping(1, os);
auto const m3 = traj.mapping(3, os);
auto const m4 = traj.mapping(4, os);
auto const m5 = traj.mapping(5, os);
auto const m6 = traj.mapping(6, os);
auto const nn = make_kernel("NN", info.type, os);
auto const kb3 = make_kernel("KB3", info.type, os);
auto const kb4 = make_kernel("KB4", info.type, os);
auto const kb5 = make_kernel("KB5", info.type, os);
auto const kb6 = make_kernel("KB6", info.type, os);

TEST_CASE("GridEcho")
{
  auto gridnn = make_grid(nn.get(), m1, false);
  auto gridkb3 = make_grid(kb3.get(), m3, false);
  auto gridkb4 = make_grid(kb4.get(), m4, false);
  auto gridkb5 = make_grid(kb5.get(), m5, false);
  auto gridkb6 = make_grid(kb6.get(), m6, false);

  BENCHMARK("NN Noncartesian->Cartesian")
  {
    gridnn->Adj(nc);
  };

  BENCHMARK("KB3 Noncartesian->Cartesian")
  {
    gridkb3->Adj(nc);
  };

  BENCHMARK("KB4 Noncartesian->Cartesian")
  {
    gridkb4->Adj(nc);
  };

  BENCHMARK("KB5 Noncartesian->Cartesian")
  {
    gridkb5->Adj(nc);
  };

  BENCHMARK("KB6 Noncartesian->Cartesian")
  {
    gridkb6->Adj(nc);
  };
}

TEST_CASE("GridBasis")
{
  Index const nB = 4;
  R2 basis(256, nB);
  basis.setConstant(1.f);

  auto gridnn = make_grid_basis(nn.get(), m1, basis, false);
  auto gridkb3 = make_grid_basis(kb3.get(), m3, basis, false);
  auto gridkb4 = make_grid_basis(kb4.get(), m4, basis, false);
  auto gridkb5 = make_grid_basis(kb5.get(), m5, basis, false);
  auto gridkb6 = make_grid_basis(kb6.get(), m6, basis, false);

  BENCHMARK("NN Noncartesian->Cartesian")
  {
    gridnn->Adj(nc);
  };

  BENCHMARK("KB3 Noncartesian->Cartesian")
  {
    gridkb3->Adj(nc);
  };

  BENCHMARK("KB4 Noncartesian->Cartesian")
  {
    gridkb4->Adj(nc);
  };

  BENCHMARK("KB5 Noncartesian->Cartesian")
  {
    gridkb5->Adj(nc);
  };

  BENCHMARK("KB6 Noncartesian->Cartesian")
  {
    gridkb6->Adj(nc);
  };
}
