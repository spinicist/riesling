#define CATCH_CONFIG_ENABLE_BENCHMARKING

#include "op/grid.hpp"
#include "info.hpp"
#include "log.hpp"
#include "traj_spirals.hpp"

#include <catch2/benchmark/catch_benchmark_all.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace rl;

Index const M = 64;
Index const C = 8;
Index const traces = M * M;
auto const  points = ArchimedeanSpiral(M, 1.f, traces);
Basis basis;
Trajectory  traj(points);
float const os = 2.f;

TEST_CASE("Grid ES3", "[grid]")
{
  Log::SetLevel(Log::Level::Testing);
  auto grid = TOps::Grid<3>::Make(traj, "ES3", os, C, &basis);
  Cx5  c(grid->ishape);
  Cx3  nc(grid->oshape);
  c.setRandom();
  nc.setRandom();
  Cx5Map  mc(c.data(), c.dimensions());
  Cx3Map  mnc(nc.data(), nc.dimensions());
  Cx5CMap cc(c.data(), c.dimensions());
  Cx3CMap cnc(nc.data(), nc.dimensions());
  BENCHMARK("forward") { grid->forward(cc, mnc); };
  BENCHMARK("iforward") { grid->iforward(cc, mnc); };
  BENCHMARK("adjoint") { grid->adjoint(cnc, mc); };
  BENCHMARK("iadjoint") { grid->iadjoint(cnc, mc); };
}

TEST_CASE("Grid ES5", "[grid]")
{
  Log::SetLevel(Log::Level::Testing);
  auto grid = TOps::Grid<3>::Make(traj, "ES5", os, C, &basis);
  Cx5  c(grid->ishape);
  Cx3  nc(grid->oshape);
  c.setRandom();
  nc.setRandom();
  Cx5Map  mc(c.data(), c.dimensions());
  Cx3Map  mnc(nc.data(), nc.dimensions());
  Cx5CMap cc(c.data(), c.dimensions());
  Cx3CMap cnc(nc.data(), nc.dimensions());
  BENCHMARK("forward") { grid->forward(cc, mnc); };
  BENCHMARK("iforward") { grid->iforward(cc, mnc); };
  BENCHMARK("adjoint") { grid->adjoint(cnc, mc); };
  BENCHMARK("iadjoint") { grid->iadjoint(cnc, mc); };
}

TEST_CASE("GridBasis ES3", "[grid]")
{
  Index const nB = 4;
  Basis basis(nB, 1, 256);
  auto grid = TOps::Grid<3>::Make(traj, "ES3", os, C, &basis);
  Cx5  c(grid->ishape);
  Cx3  nc(grid->oshape);
  c.setRandom();
  nc.setRandom();
  Cx5Map  mc(c.data(), c.dimensions());
  Cx3Map  mnc(nc.data(), nc.dimensions());
  Cx5CMap cc(c.data(), c.dimensions());
  Cx3CMap cnc(nc.data(), nc.dimensions());
  BENCHMARK("forward") { grid->forward(cc, mnc); };
  BENCHMARK("iforward") { grid->iforward(cc, mnc); };
  BENCHMARK("adjoint") { grid->adjoint(cnc, mc); };
  BENCHMARK("iadjoint") { grid->iadjoint(cnc, mc); };
}

TEST_CASE("GridBasis ES5", "[grid]")
{
  Index const nB = 4;
  Basis basis(nB, 1, 256);
  auto grid = TOps::Grid<3>::Make(traj, "ES5", os, C, &basis);
  Cx5  c(grid->ishape);
  Cx3  nc(grid->oshape);
  c.setRandom();
  nc.setRandom();
  Cx5Map  mc(c.data(), c.dimensions());
  Cx3Map  mnc(nc.data(), nc.dimensions());
  Cx5CMap cc(c.data(), c.dimensions());
  Cx3CMap cnc(nc.data(), nc.dimensions());
  BENCHMARK("forward") { grid->forward(cc, mnc); };
  BENCHMARK("iforward") { grid->iforward(cc, mnc); };
  BENCHMARK("adjoint") { grid->adjoint(cnc, mc); };
  BENCHMARK("iadjoint") { grid->iadjoint(cnc, mc); };
}
