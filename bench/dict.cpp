#define CATCH_CONFIG_ENABLE_BENCHMARKING

#include "func/dict.hpp"
#include "algo/decomp.hpp"
#include "sim/t2flair.hpp"
#include "threads.hpp"
#include "tensorOps.hpp"

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

  // Calculate SVD - observations are in rows
  auto const svd = rl::SVD<float>(dynamics, true, false);
  Index const nRetain = 3;
  Eigen::MatrixXf basis = svd.V.leftCols(nRetain).array();
  Eigen::ArrayXf const scales = svd.vals.head(nRetain) / svd.vals(0);
  Eigen::MatrixXf dict = basis.transpose() * dynamics.matrix();
  Eigen::ArrayXf const norm = dict.colwise().norm();
  dict = dict.array().rowwise() / norm.transpose();

  rl::BruteForceDictionary brute(dict);
  rl::BallTreeDictionary ball(dict);

  BENCHMARK("Brute-Force Lookup")
  {
    brute.project(dict.col(nsamp/2));
  };

  BENCHMARK("Ball-Tree Lookup")
  {
    ball.project(dict.col(nsamp/2));
  };
}
