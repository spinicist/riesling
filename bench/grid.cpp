#define CATCH_CONFIG_ENABLE_BENCHMARKING

#include "../src/info.h"
#include "../src/op/grid-internal.hpp"
#include "../src/traj_spirals.h"

#include <catch2/catch.hpp>

using namespace rl;

Index const M = 128;
Index const C = 8;
Info const info{.channels = C, .samples = M / 2, .traces = Index(M * M), .matrix = Sz3{M, M, M}};
auto const points = ArchimedeanSpiral(info.samples, info.traces);
Trajectory traj(info, points);
float const os = 2.f;
Index const bucketSz = 32;
auto const nn = make_kernel("NN", true, os);
auto const kb3 = make_kernel("KB3", true, os);
auto const kb5 = make_kernel("KB5", true, os);
auto const fi3 = make_kernel("FI3", true, os);
auto const fi5 = make_kernel("FI5", true, os);

Mapping const m1(traj, nn.get(), os, bucketSz);
Mapping const m3(traj, kb3.get(), os, bucketSz);
Mapping const m5(traj, kb5.get(), os, bucketSz);

TEST_CASE("GridAdj")
{
  auto gridnn = make_grid_internal<1, 1, Cx>(nn.get(), m1, C);
  auto gridkb3 = make_grid_internal<3, 3, Cx>(kb3.get(), m3, C);
  auto gridkb5 = make_grid_internal<5, 5, Cx>(kb5.get(), m5, C);
  auto gridfi3 = make_grid_internal<3, 3, Cx>(fi3.get(), m3, C);
  auto gridfi5 = make_grid_internal<5, 5, Cx>(fi5.get(), m5, C);

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
  auto gridnn = make_grid_internal<1, 1, Cx>(nn.get(), m1, C);
  auto gridkb3 = make_grid_internal<3, 3, Cx>(kb3.get(), m3, C);
  auto gridkb5 = make_grid_internal<5, 5, Cx>(kb5.get(), m5, C);
  auto gridfi3 = make_grid_internal<3, 3, Cx>(fi3.get(), m3, C);
  auto gridfi5 = make_grid_internal<5, 5, Cx>(fi5.get(), m5, C);

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
  Re2 basis(256, nB);
  basis.setConstant(1.f);

  auto gridnn = make_grid_internal<1, 1, Cx>(nn.get(), m1, C, basis);
  auto gridkb3 = make_grid_internal<3, 3, Cx>(kb3.get(), m3, C, basis);
  auto gridkb5 = make_grid_internal<5, 5, Cx>(kb5.get(), m5, C, basis);
  auto gridfi3 = make_grid_internal<3, 3, Cx>(fi3.get(), m3, C, basis);
  auto gridfi5 = make_grid_internal<5, 5, Cx>(fi5.get(), m5, C, basis);

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
  Re2 basis(256, nB);
  basis.setConstant(1.f);

  auto gridnn = make_grid_internal<1, 1, Cx>(nn.get(), m1, C, basis);
  auto gridkb3 = make_grid_internal<3, 3, Cx>(kb3.get(), m3, C, basis);
  auto gridkb5 = make_grid_internal<5, 5, Cx>(kb5.get(), m5, C, basis);
  auto gridfi3 = make_grid_internal<3, 3, Cx>(fi3.get(), m3, C, basis);
  auto gridfi5 = make_grid_internal<5, 5, Cx>(fi5.get(), m5, C, basis);

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
