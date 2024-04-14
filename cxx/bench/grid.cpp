#define CATCH_CONFIG_ENABLE_BENCHMARKING

#include "info.hpp"
#include "op/grid.hpp"
#include "traj_spirals.h"

#include <catch2/benchmark/catch_benchmark_all.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace rl;

Index const M = 64;
Index const C = 8;
Index const samples = M / 2;
Index const traces = M * M;
Info const  info{.matrix = Sz3{M, M, M}};
auto const  points = ArchimedeanSpiral(samples, traces);
Trajectory  traj(info, points);
float const os = 2.f;

TEST_CASE("Grid ES3", "[grid]")
{
  Log::SetLevel(Log::Level::Testing);
  auto gridfi3 = Grid<Cx, 3>::Make(traj, "ES3", os, C);
  Cx5  c(gridfi3->ishape);
  Cx3  nc(gridfi3->oshape);
  nc.setRandom();
  BENCHMARK("ES3 Noncartesian->Cartesian") { gridfi3->adjoint(nc); };
  BENCHMARK("ES3 Cartesian->Noncartesian") { gridfi3->forward(c); };
}

TEST_CASE("Grid ES5", "[grid]")
{
  Log::SetLevel(Log::Level::Testing);
  auto gridfi5 = Grid<Cx, 3>::Make(traj, "ES5", os, C);
  Cx5  c(gridfi5->ishape);
  Cx3  nc(gridfi5->oshape);
  nc.setRandom();
  BENCHMARK("ES5 Noncartesian->Cartesian") { gridfi5->adjoint(nc); };
  BENCHMARK("ES5 Cartesian->Noncartesian") { gridfi5->forward(c); };
}

TEST_CASE("GridBasis ES3", "[grid]")
{
  Index const nB = 4;
  Re2         basis(nB, 256);
  basis.setConstant(1.f);
  auto gridfi3 = Grid<Cx, 3>::Make(traj, "ES3", os, C, basis);
  Cx5  c(gridfi3->ishape);
  Cx3  nc(gridfi3->oshape);
  BENCHMARK("ES3 Noncartesian->Cartesian") { gridfi3->adjoint(nc); };
  BENCHMARK("ES3 Cartesian->Noncartesian") { gridfi3->forward(c); };
}

TEST_CASE("GridBasis ES5", "[grid]")
{
  Index const nB = 4;
  Re2         basis(nB, 256);
  basis.setConstant(1.f);
  auto gridfi5 = Grid<Cx, 3>::Make(traj, "ES5", os, C, basis);
  Cx5  c(gridfi5->ishape);
  Cx3  nc(gridfi5->oshape);
  BENCHMARK("ES5 Noncartesian->Cartesian") { gridfi5->adjoint(nc); };
  BENCHMARK("ES5 Cartesian->Noncartesian") { gridfi5->forward(c); };
}
