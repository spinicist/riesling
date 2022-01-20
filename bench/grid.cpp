#define CATCH_CONFIG_ENABLE_BENCHMARKING

#include "../src/op/grid.h"
#include "../src/info.h"
#include "../src/traj_spirals.h"

#include <catch2/catch.hpp>

TEST_CASE("Grid")
{
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
  auto const points = ArchimedeanSpiral(info.read_points, info.spokes);
  Trajectory traj(info, points);

  float const os = 2.f;

  BENCHMARK("Mapping")
  {
    traj.mapping(1, os);
  };

  auto const nn = make_kernel("NN", info.type, os);
  auto const kb = make_kernel("KB5", info.type, os);

  auto const m1 = traj.mapping(1, os);
  auto const m5 = traj.mapping(5, os);

  auto gridnn = make_grid(nn.get(), m1, false);
  auto gridkb = make_grid(kb.get(), m5, false);

  auto nc = info.noncartesianVolume();
  Cx5 c(gridkb->inputDimensions(C));
  Index const nB = 4;
  R2 basis(256, nB);
  basis.setConstant(1.f);

  auto gridbnn = make_grid_basis(nn.get(), m1, basis, false);
  auto gridbkb = make_grid_basis(kb.get(), m5, basis, false);
  Cx5 b(gridbkb->inputDimensions(C));

  BENCHMARK("NN Noncartesian->Cartesian")
  {
    gridnn->Adj(nc, c);
  };
  BENCHMARK("NN Basis Noncartesian->Cartesian")
  {
    gridbnn->Adj(nc, b);
  };
  BENCHMARK("KB Noncartesian->Cartesian")
  {
    gridkb->Adj(nc, c);
  };
  BENCHMARK("KB Basis Noncartesian->Cartesian")
  {
    gridbkb->Adj(nc, b);
  };

  BENCHMARK("NN Cartesian->Noncartesian")
  {
    gridnn->A(c, nc);
  };
  BENCHMARK("KB Cartesian->Noncartesian")
  {
    gridkb->A(c, nc);
  };
  BENCHMARK("NN Basis Cartesian->Noncartesian")
  {
    gridbnn->A(b, nc);
  };
  BENCHMARK("KB Basis Cartesian->Noncartesian")
  {
    gridbkb->A(b, nc);
  };
}
