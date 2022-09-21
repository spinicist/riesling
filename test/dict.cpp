#define CATCH_CONFIG_ENABLE_BENCHMARKING

#include "func/dict.hpp"
#include "algo/decomp.hpp"
#include "sim/t2flair.hpp"
#include "tensorOps.hpp"
#include "threads.hpp"

#include <catch2/benchmark/catch_benchmark_all.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace Catch;

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
  for (Index ii = 0; ii < parameters.cols(); ii++) {
    dynamics.col(ii) = simulator.simulate(parameters.col(ii));
  }

  // Calculate SVD - observations are in columns so transpose
  auto const svd = rl::SVD<float>(dynamics, true, false);
  Index const nRetain = 3;
  Eigen::MatrixXf basis = svd.V.leftCols(nRetain).array();
  Eigen::ArrayXf const scales = svd.vals.head(nRetain) / svd.vals(0);
  Eigen::MatrixXf dict = basis.transpose() * dynamics.matrix();
  Eigen::ArrayXf const norm = dict.colwise().norm();
  dict = dict.array().rowwise() / norm.transpose();

  // Should get back exact matches within precision
  SECTION("Brute-Force Lookup")
  {
    rl::BruteForceDictionary brute(dict);
    CHECK((dict.col(0) - brute.project(dict.col(0))).norm() == Approx(0.f).margin(1.e-6f));
    CHECK((dict.col(nsamp / 2) - brute.project(dict.col(nsamp / 2))).norm() == Approx(0.f).margin(1.e-6f));
  };

  SECTION("Ball-Tree Lookup")
  {
    rl::BallTreeDictionary ball(dict);
    CHECK((dict.col(0) - ball.project(dict.col(0))).norm() == Approx(0.f).margin(1.e-6f));
    CHECK((dict.col(nsamp / 2) - ball.project(dict.col(nsamp / 2))).norm() == Approx(0.f).margin(1.e-6f));
  };
}
