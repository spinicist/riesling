#define CATCH_CONFIG_ENABLE_BENCHMARKING

#include "algo/common.hpp"
#include <catch2/benchmark/catch_benchmark_all.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace Catch;

TEST_CASE("Norms", "[Norm]")
{
  int const        sz = 256 * 256 * 256 * 16;
  Eigen::VectorXcf x(sz);
  x.setRandom();

  float sn = 0.f, n = 0.f, pn = 0.f;

  BENCHMARK("Stable Norm") { sn = x.stableNorm(); };
  BENCHMARK("Norm") { n = x.norm(); };
  BENCHMARK("Parallel Norm") { pn = rl::ParallelNorm(x); };

  INFO("Stable norm   " << sn << 
     "\nNorm          " << n << 
     "\nParallel Norm " << pn <<
     "\nThreshold     " << 1.e-6f * sz);
  CHECK(std::abs(n - sn) == Approx(0.f).margin(1.e-6f * sz));
  CHECK(std::abs(pn - sn) == Approx(0.f).margin(1.e-6f * sz));
}
