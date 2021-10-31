#define CATCH_CONFIG_ENABLE_BENCHMARKING

#include "../src/info.h"
#include "../src/op/grid-kb.h"
#include "../src/op/grid-nn.h"
#include "../src/op/grid.h"
#include "../src/traj_spirals.h"

#include <catch2/catch.hpp>

TEST_CASE("Grid-NN")
{
  Log log;
  long const M = 64;
  long const C = 8;
  Info const info{.type = Info::Type::ThreeD,
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
  auto nc = info.noncartesianVolume();
  auto c = gridnn.newMultichannel(C);

  BENCHMARK("NN Noncartesian->Cartesian")
  {
    gridnn.Adj(nc, c);
  };
  BENCHMARK("NN Cartesian->Noncartesian")
  {
    gridnn.A(c, nc);
  };
}

TEST_CASE("Grid-KB")
{
  Log log;
  long const M = 64;
  long const C = 8;
  Info const info{.type = Info::Type::ThreeD,
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

  GridKB3D gridkb(traj, os, false, log);
  auto nc = info.noncartesianVolume();
  auto c = gridkb.newMultichannel(C);

  BENCHMARK("KB Noncartesian->Cartesian")
  {
    gridkb.Adj(nc, c);
  };
  BENCHMARK("KB Cartesian->Noncartesian")
  {
    gridkb.A(c, nc);
  };
}