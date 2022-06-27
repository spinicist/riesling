#define CATCH_CONFIG_ENABLE_BENCHMARKING

#include "../src/info.h"
#include "../src/traj_spirals.h"
#include "../src/op/grid-base.hpp"

#include <catch2/catch.hpp>

// Forward declarations
std::unique_ptr<GridBase> make_1(Kernel const *, Mapping const &, Index const nC);
std::unique_ptr<GridBase> make_1_b(Kernel const *, Mapping const &, Index const nC, R2 const &b);
std::unique_ptr<GridBase> make_3(Kernel const *, Mapping const &, Index const nC);
std::unique_ptr<GridBase> make_3_b(Kernel const *, Mapping const &, Index const nC, R2 const &b);
std::unique_ptr<GridBase> make_5(Kernel const *, Mapping const &, Index const nC);
std::unique_ptr<GridBase> make_5_b(Kernel const *, Mapping const &, Index const nC, R2 const &b);

Index const M = 128;
Index const C = 8;
Info const info{
  .type = Info::Type::ThreeD,
  .matrix = Eigen::Array3l::Constant(M),
  .channels = C,
  .read_points = M / 2,
  .spokes = M * M,
  .volumes = 1,
  .frames = 1,
  .tr = 1.f,
  .voxel_size = Eigen::Array3f::Constant(1.f),
  .origin = Eigen::Array3f::Constant(0.f),
  .direction = Eigen::Matrix3f::Identity()};
auto const points = ArchimedeanSpiral(info.read_points, info.spokes);
Trajectory traj(info, points);
float const os = 2.f;
Index const bucketSz = 32;
auto const nn = make_kernel("NN", info.type, os);
auto const kb3 = make_kernel("KB3", info.type, os);
auto const kb5 = make_kernel("KB5", info.type, os);
auto const fi3 = make_kernel("FI3", info.type, os);
auto const fi5 = make_kernel("FI5", info.type, os);

Mapping const m1(traj, nn.get(), os, bucketSz);
Mapping const m3(traj, kb3.get(), os, bucketSz);
Mapping const m5(traj, kb5.get(), os, bucketSz);

TEST_CASE("GridAdj")
{
  auto gridnn = make_1(nn.get(), m1, C);
  auto gridkb3 = make_3(kb3.get(), m3, C);
  auto gridkb5 = make_5(kb5.get(), m5, C);
  auto gridfi3 = make_3(fi3.get(), m3, C);
  auto gridfi5 = make_5(fi5.get(), m5, C);

  Cx3 nc(gridnn->outputDimensions());
  nc.setRandom();

  BENCHMARK("NN Noncartesian->Cartesian")
  {
    gridnn->Adj(nc);
  };

  BENCHMARK("KB3 Noncartesian->Cartesian")
  {
    gridkb3->Adj(nc);
  };

  BENCHMARK("KB5 Noncartesian->Cartesian")
  {
    gridkb5->Adj(nc);
  };

  BENCHMARK("FI3 Noncartesian->Cartesian")
  {
    gridfi3->Adj(nc);
  };

  BENCHMARK("FI5 Noncartesian->Cartesian")
  {
    gridfi5->Adj(nc);
  };
}

TEST_CASE("GridA")
{
  auto gridnn = make_1(nn.get(), m1, C);
  auto gridkb3 = make_3(kb3.get(), m3, C);
  auto gridkb5 = make_5(kb5.get(), m5, C);
  auto gridfi3 = make_3(fi3.get(), m3, C);
  auto gridfi5 = make_5(fi5.get(), m5, C);

  Cx5 c(gridnn->inputDimensions());

  BENCHMARK("NN Cartesian->Noncartesian")
  {
    gridnn->A(c);
  };

  BENCHMARK("KB3 Cartesian->Noncartesian")
  {
    gridkb3->A(c);
  };

  BENCHMARK("KB5 Cartesian->Noncartesian")
  {
    gridkb5->A(c);
  };

  BENCHMARK("FI3 Cartesian->Noncartesian")
  {
    gridfi3->A(c);
  };

  BENCHMARK("FI5 Cartesian->Noncartesian")
  {
    gridfi5->A(c);
  };
}

TEST_CASE("GridBasisAdj")
{
  Index const nB = 4;
  R2 basis(256, nB);
  basis.setConstant(1.f);

  auto gridnn = make_1_b(nn.get(), m1, C, basis);
  auto gridkb3 = make_3_b(kb3.get(), m3, C, basis);
  auto gridkb5 = make_5_b(kb5.get(), m5, C, basis);
  auto gridfi3 = make_3_b(fi3.get(), m3, C, basis);
  auto gridfi5 = make_5_b(fi5.get(), m5, C, basis);

  Cx3 nc(gridnn->outputDimensions());

  BENCHMARK("NN Noncartesian->Cartesian")
  {
    gridnn->Adj(nc);
  };

  BENCHMARK("KB3 Noncartesian->Cartesian")
  {
    gridkb3->Adj(nc);
  };

  BENCHMARK("KB5 Noncartesian->Cartesian")
  {
    gridkb5->Adj(nc);
  };

  BENCHMARK("FI3 Noncartesian->Cartesian")
  {
    gridfi3->Adj(nc);
  };

  BENCHMARK("FI5 Noncartesian->Cartesian")
  {
    gridfi5->Adj(nc);
  };
}

TEST_CASE("GridBasisA")
{
  Index const nB = 4;
  R2 basis(256, nB);
  basis.setConstant(1.f);

  auto gridnn = make_1_b(nn.get(), m1, C, basis);
  auto gridkb3 = make_3_b(kb3.get(), m3, C, basis);
  auto gridkb5 = make_5_b(kb5.get(), m5, C, basis);
  auto gridfi3 = make_3_b(fi3.get(), m3, C, basis);
  auto gridfi5 = make_5_b(fi5.get(), m5, C, basis);

  Cx5 c(gridnn->inputDimensions());

  BENCHMARK("NN Cartesian->Noncartesian")
  {
    gridnn->A(c);
  };

  BENCHMARK("KB3 Cartesian->Noncartesian")
  {
    gridkb3->A(c);
  };

  BENCHMARK("KB5 Cartesian->Noncartesian")
  {
    gridkb5->A(c);
  };

  BENCHMARK("FI3 Cartesian->Noncartesian")
  {
    gridfi3->A(c);
  };

  BENCHMARK("FI5 Cartesian->Noncartesian")
  {
    gridfi5->A(c);
  };
}
