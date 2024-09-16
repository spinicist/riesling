#define CATCH_CONFIG_ENABLE_BENCHMARKING

#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark_all.hpp>
#include "sys/threads.hpp"

TEST_CASE("Tensor Dot", "[Dot]")
{
  using Cx4 = Eigen::Tensor<std::complex<float>, 4>;
  int const sz = 256;
  Cx4 grid(sz, sz, sz, 16);
  grid.setRandom();

  Eigen::TensorFixedSize<std::complex<float>, Eigen::Sizes<>> d;
  BENCHMARK("Naive")
  {
    d = (grid.conjugate() * grid).sum();
  };

  BENCHMARK("Threaded")
  {
    d.device(rl::Threads::TensorDevice()) = (grid.conjugate() * grid).sum();
  };
}

TEST_CASE("Core Dot", "[Dot]")
{
  int const sz = 256*256*256*16;
  Eigen::VectorXcf vec1(sz), vec2(sz); 
  vec1.setRandom();
  vec2.setRandom();

  rl::Cx d;
  BENCHMARK("Dot")
  {
    d = vec1.dot(vec2);
  };

  float n;

  BENCHMARK("Norm")
  {
    n = vec1.norm();
  };

  BENCHMARK("Stable Norm")
  {
    n = vec1.stableNorm();
  };
}
