#define CATCH_CONFIG_ENABLE_BENCHMARKING

#include "op/grid.hpp"
#include "info.hpp"
#include "log.hpp"
#include "traj_spirals.hpp"

#include <catch2/benchmark/catch_benchmark_all.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

using namespace rl;

Index const M = 64;
Index const C = 8;
Index const traces = M * M;
auto const  points = ArchimedeanSpiral(M, 1.f, traces);
Trajectory  traj(points);
float const os = 2.f;

TEST_CASE("Grid", "[grid]")
{
  auto ktype = GENERATE("ES2", "ES4", "ES6");
  Log::SetLevel(Log::Level::Testing);
  auto grid = TOps::Grid<3>::Make(traj, traj.matrix(), os, ktype, C, nullptr);
  Cx5  c(grid->ishape);
  Cx3  nc(grid->oshape);
  c.setRandom();
  nc.setRandom();
  Cx5Map  mc(c.data(), c.dimensions());
  Cx3Map  mnc(nc.data(), nc.dimensions());
  Cx5CMap cc(c.data(), c.dimensions());
  Cx3CMap cnc(nc.data(), nc.dimensions());
  BENCHMARK(fmt::format("Grid {} forward", ktype)) { grid->forward(cc, mnc); };
  BENCHMARK(fmt::format("Grid {} iforward", ktype)) { grid->iforward(cc, mnc); };
  BENCHMARK(fmt::format("Grid {} adjoint", ktype)) { grid->adjoint(cnc, mc); };
  BENCHMARK(fmt::format("Grid {} iadjoint", ktype)) { grid->iadjoint(cnc, mc); };
}

TEST_CASE("Grid-Basis", "[grid]")
{
  auto ktype = GENERATE("ES2", "ES4", "ES6");
  Index const nB = 4;
  Basis       basis(nB, 1, 256);
  auto        grid = TOps::Grid<3>::Make(traj, traj.matrix(), os, ktype, C, &basis);
  Cx5         c(grid->ishape);
  Cx3         nc(grid->oshape);
  c.setRandom();
  nc.setRandom();
  Cx5Map  mc(c.data(), c.dimensions());
  Cx3Map  mnc(nc.data(), nc.dimensions());
  Cx5CMap cc(c.data(), c.dimensions());
  Cx3CMap cnc(nc.data(), nc.dimensions());
  BENCHMARK(fmt::format("Grid {} forward", ktype)) { grid->forward(cc, mnc); };
  BENCHMARK(fmt::format("Grid {} iforward", ktype)) { grid->iforward(cc, mnc); };
  BENCHMARK(fmt::format("Grid {} adjoint", ktype)) { grid->adjoint(cnc, mc); };
  BENCHMARK(fmt::format("Grid {} iadjoint", ktype)) { grid->iadjoint(cnc, mc); };
}
