#define CATCH_CONFIG_ENABLE_BENCHMARKING

#include "../src/gridder.h"
#include "../src/info.h"
#include "../src/kernel_kb.h"
#include "../src/traj_archimedean.h"
#include <catch2/catch.hpp>

TEST_CASE("Grid")
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
  KaiserBessel kernel(3, os, true);

  BENCHMARK("Mapping")
  {
    traj.mapping(os, kernel.radius());
  };

  Gridder gridder(traj.mapping(os, kernel.radius()), &kernel, false, log);
  auto nc = info.noncartesianVolume();
  auto c = gridder.newMultichannel(C);

  BENCHMARK("Noncartesian->Cartesian")
  {
    gridder.toCartesian(nc, c);
  };
  BENCHMARK("Cartesian->Noncartesian")
  {
    gridder.toNoncartesian(c, nc);
  };
}