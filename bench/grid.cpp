#define CATCH_CONFIG_ENABLE_BENCHMARKING

#include "../src/op/grid.h"
#include "../src/info.h"
#include "../src/op/grid-basis-kb.h"
#include "../src/op/grid-basis-nn.h"
#include "../src/op/grid-kb.h"
#include "../src/op/grid-nn.h"
#include "../src/traj_spirals.h"

#include <catch2/catch.hpp>

TEST_CASE("Grid")
{
  Log log;
  long const M = 32;
  long const C = 8;
  Info const info{
    .type = Info::Type::ThreeD,
    .channels = C,
    .matrix = Eigen::Array3l::Constant(M),
    .read_points = M / 2,
    .read_gap = 0,
    .spokes_hi = M * M,
    .spokes_lo = 0,
    .lo_scale = 1.f,
    .volumes = 1,
    .echoes = 1,
    .tr = 1.f,
    .voxel_size = Eigen::Array3f::Constant(1.f),
    .origin = Eigen::Array3f::Constant(0.f),
    .direction = Eigen::Matrix3f::Identity()};
  auto const points = ArchimedeanSpiral(info);
  Trajectory traj(info, points, log);

  float const os = 2.f;

  BENCHMARK("Mapping")
  {
    traj.mapping(os, 1);
  };

  GridNN gridnn(traj, os, false, log);
  GridKB<5, 5> gridkb(traj, os, false, log);

  auto nc = info.noncartesianVolume();
  auto c = gridkb.newMultichannel(C);
  long const nB = 4;
  R2 basis(256, nB);
  basis.setConstant(1.f);

  GridBasisNN gridbnn(traj, os, false, basis, log);
  GridBasisKB<5, 5> gridbkb(traj, os, false, basis, log);
  Cx5 b(C, nB, c.dimension(1), c.dimension(2), c.dimension(3));

  BENCHMARK("NN Noncartesian->Cartesian")
  {
    gridnn.Adj(nc, c);
  };
  BENCHMARK("NN Basis Noncartesian->Cartesian")
  {
    gridbnn.Adj(nc, b);
  };
  BENCHMARK("KB Noncartesian->Cartesian")
  {
    gridkb.Adj(nc, c);
  };
  BENCHMARK("KB Basis Noncartesian->Cartesian")
  {
    gridbkb.Adj(nc, b);
  };

  BENCHMARK("NN Cartesian->Noncartesian")
  {
    gridnn.A(c, nc);
  };
  BENCHMARK("KB Cartesian->Noncartesian")
  {
    gridkb.A(c, nc);
  };
  BENCHMARK("NN Basis Cartesian->Noncartesian")
  {
    gridbnn.A(b, nc);
  };
  BENCHMARK("KB Basis Cartesian->Noncartesian")
  {
    gridbkb.A(b, nc);
  };
}