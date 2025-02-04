#define CATCH_CONFIG_ENABLE_BENCHMARKING

#include "rl/op/grid.hpp"
#include "rl/info.hpp"
#include "rl/log.hpp"
#include "rl/phantom/radial.hpp"

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
  auto grid = TOps::Grid<3>(TOps::Grid<3>::Opts{.osamp = os}, traj, C, nullptr);
  Cx5  c(grid.ishape);
  Cx3  nc(grid.oshape);
  c.setRandom();
  nc.setRandom();
  Cx5Map  mc(c.data(), c.dimensions());
  Cx3Map  mnc(nc.data(), nc.dimensions());
  Cx5CMap cc(c.data(), c.dimensions());
  Cx3CMap cnc(nc.data(), nc.dimensions());
  BENCHMARK(fmt::format("Grid forward")) { grid.forward(cc, mnc); };
  BENCHMARK(fmt::format("Grid iforward")) { grid.iforward(cc, mnc); };
  BENCHMARK(fmt::format("Grid adjoint")) { grid.adjoint(cnc, mc); };
  BENCHMARK(fmt::format("Grid iadjoint")) { grid.iadjoint(cnc, mc); };
}

TEST_CASE("Grid-Basis", "[grid]")
{
  Index const nB = 4;
  Basis       basis(nB, 1, 256);
  auto        grid = TOps::Grid<3>(TOps::Grid<3>::Opts{.osamp = os}, traj, C, &basis);
  Cx5         c(grid.ishape);
  Cx3         nc(grid.oshape);
  c.setRandom();
  nc.setRandom();
  Cx5Map  mc(c.data(), c.dimensions());
  Cx3Map  mnc(nc.data(), nc.dimensions());
  Cx5CMap cc(c.data(), c.dimensions());
  Cx3CMap cnc(nc.data(), nc.dimensions());
  BENCHMARK(fmt::format("Grid forward")) { grid.forward(cc, mnc); };
  BENCHMARK(fmt::format("Grid iforward")) { grid.iforward(cc, mnc); };
  BENCHMARK(fmt::format("Grid adjoint")) { grid.adjoint(cnc, mc); };
  BENCHMARK(fmt::format("Grid iadjoint")) { grid.iadjoint(cnc, mc); };
}
