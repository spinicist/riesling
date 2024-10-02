#define CATCH_CONFIG_ENABLE_BENCHMARKING

#include "algo/common.hpp"
#include <catch2/benchmark/catch_benchmark_all.hpp>
#include <catch2/catch_test_macros.hpp>

TEST_CASE("Norms", "[Norm]")
{
  int const        sz = 256 * 256 * 256 * 16;
  Eigen::VectorXcf a(sz);
  a.setRandom();
  float n;

  BENCHMARK("ParNorm")
  {
    n = rl::ParNorm(a);
  };
  
  BENCHMARK(".norm")
  {
    n = a.norm();
  };
}
