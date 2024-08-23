#define CATCH_CONFIG_ENABLE_BENCHMARKING

#include "op/nufft.hpp"
#include "info.hpp"
#include "log.hpp"
#include "traj_spirals.hpp"

#include <catch2/benchmark/catch_benchmark_all.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace rl;

namespace {
Index const M = 64;
Index const C = 8;
Index const traces = M * M;
auto const  points = ArchimedeanSpiral(M, 1.f, traces);
Basis       basis;
Trajectory  traj(points);
float const os = 2.f;
} // namespace

TEST_CASE("NUFFT", "[nufft]")
{
  Log::SetLevel(Log::Level::Testing);
  TOps::NUFFT<3> nufft(traj, "ES3", os, C, &basis);
  Cx5            c(nufft.ishape);
  Cx3            nc(nufft.oshape);
  c.setRandom();
  nc.setRandom();
  Cx5Map  mc(c.data(), c.dimensions());
  Cx3Map  mnc(nc.data(), nc.dimensions());
  Cx5CMap cc(c.data(), c.dimensions());
  Cx3CMap cnc(nc.data(), nc.dimensions());
  BENCHMARK("forward") { nufft.forward(cc, mnc); };
  BENCHMARK("iforward") { nufft.iforward(cc, mnc); };
  BENCHMARK("adjoint") { nufft.adjoint(cnc, mc); };
  BENCHMARK("iadjoint") { nufft.iadjoint(cnc, mc); };
}

TEST_CASE("NUFFT Basis", "[nufft]")
{
  Index const    nB = 4;
  Basis          basis(nB, 1, 256);
  TOps::NUFFT<3> nufft(traj, "ES3", os, C, &basis);
  Cx5            c(nufft.ishape);
  Cx3            nc(nufft.oshape);
  c.setRandom();
  nc.setRandom();
  Cx5Map  mc(c.data(), c.dimensions());
  Cx3Map  mnc(nc.data(), nc.dimensions());
  Cx5CMap cc(c.data(), c.dimensions());
  Cx3CMap cnc(nc.data(), nc.dimensions());
  BENCHMARK("forward") { nufft.forward(cc, mnc); };
  BENCHMARK("iforward") { nufft.iforward(cc, mnc); };
  BENCHMARK("adjoint") { nufft.adjoint(cnc, mc); };
  BENCHMARK("iadjoint") { nufft.iadjoint(cnc, mc); };
}
