#define CATCH_CONFIG_ENABLE_BENCHMARKING

#include "basis/basis.hpp"
#include "func/dict.hpp"
#include "algo/decomp.hpp"
#include "basis/basis.hpp"
#include "sim/t2flair.hpp"
#include "tensorOps.hpp"
#include "threads.hpp"

#include <catch2/benchmark/catch_benchmark_all.hpp>
#include <catch2/catch_test_macros.hpp>

TEST_CASE("Dictionaries", "[dict]")
{
  rl::Log::SetLevel(rl::Log::Level::Testing);

  rl::Settings const settings{
    .spg = 256,
    .gps = 6,
    .gprep2 = 2,
    .alpha = 2.f,
    .ascale = 1.f,
    .TR = 2.e-3f,
    .Tramp = 10.e-3f,
    .Tssi = 10.e-3f,
    .TI = 50.e-3f,
    .Trec = 0.e-3f,
    .TE = 80.e-3f};
  Index const nsamp = 8192;

  rl::T2FLAIR simulator{settings};
  Eigen::ArrayXXf parameters = simulator.parameters(nsamp);
  Eigen::ArrayXXf dynamics(simulator.length(), parameters.cols());
  auto task = [&](Index const ii) { dynamics.col(ii) = simulator.simulate(parameters.col(ii)); };
  rl::Threads::For(task, parameters.cols(), "Simulation");

  rl::Basis basis(parameters, dynamics, 0.f, 3, false);

  rl::BruteForceDictionary brute(basis.dict);
  rl::BallTreeDictionary ball(basis.dict);

  BENCHMARK("Brute-Force Lookup")
  {
    brute.project(basis.dict.col(nsamp / 2));
  };

  BENCHMARK("Ball-Tree Lookup")
  {
    ball.project(basis.dict.col(nsamp / 2));
  };
}
