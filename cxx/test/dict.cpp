#define CATCH_CONFIG_ENABLE_BENCHMARKING

#include "func/dict.hpp"
#include "algo/decomp.hpp"
#include "basis/basis.hpp"
#include "sim/t2flair.hpp"
#include "tensors.hpp"
#include "sys/threads.hpp"

#include <catch2/benchmark/catch_benchmark_all.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace Catch;

TEST_CASE("Dictionaries", "[dict]")
{
  rl::Log::SetLevel(rl::Log::Level::Testing);

  rl::Pars const p{
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

  rl::T2FLAIR simulator{p};
  Eigen::ArrayXXf parameters = simulator.parameters(nsamp);
  Eigen::ArrayXXf dynamics(simulator.length(), parameters.cols());
  for (Index ii = 0; ii < parameters.cols(); ii++) {
    dynamics.col(ii) = simulator.simulate(parameters.col(ii));
  }

  rl::Basis basis(parameters, dynamics, 0.f, 3, false);

  // Should get back exact matches within precision
  SECTION("Brute-Force Lookup")
  {
    rl::BruteForceDictionary brute(basis.dict);
    CHECK((basis.dict.col(0) - brute.project(basis.dict.col(0))).norm() == Approx(0.f).margin(1.e-6f));
    CHECK((basis.dict.col(nsamp / 2) - brute.project(basis.dict.col(nsamp / 2))).norm() == Approx(0.f).margin(1.e-6f));
  };

  SECTION("Ball-Tree Lookup")
  {
    rl::BallTreeDictionary ball(basis.dict);
    CHECK((basis.dict.col(0) - ball.project(basis.dict.col(0))).norm() == Approx(0.f).margin(1.e-6f));
    CHECK((basis.dict.col(nsamp / 2) - ball.project(basis.dict.col(nsamp / 2))).norm() == Approx(0.f).margin(1.e-6f));
  };
}
