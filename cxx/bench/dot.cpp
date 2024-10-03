#define CATCH_CONFIG_ENABLE_BENCHMARKING

#include "algo/common.hpp"
#include "sys/threads.hpp"
#include <catch2/benchmark/catch_benchmark_all.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace Catch;

TEST_CASE("Tensor-Dot", "[Dot]")
{
  using Cx4 = Eigen::Tensor<std::complex<float>, 4>;
  int const sz = 256;
  Cx4       a(sz, sz, sz, 16), b(sz, sz, sz, 16);
  a.setRandom();
  b.setRandom();

  Eigen::TensorFixedSize<std::complex<float>, Eigen::Sizes<>> d;
  BENCHMARK("Dot") { d = (a * b.conjugate()).sum(); };

  BENCHMARK("Threaded Dot") { d.device(rl::Threads::TensorDevice()) = (a * b.conjugate()).sum(); };

  float n;
  BENCHMARK("Norm")
  {
    d = (a * b.conjugate()).sum();
    n = std::sqrt(std::abs(d()));
  };
}

TEST_CASE("Core-Dot", "[Dot]")
{
  int const        sz = 256 * 256 * 256 * 16;
  Eigen::VectorXcf vec1(sz), vec2(sz);
  vec1.setRandom();
  vec2.setRandom();

  rl::Cx d = 0.f, pwd = 0.f, pd = 0.f;

  BENCHMARK("Dot") { d = vec1.dot(vec2); };
  BENCHMARK("Pairwise Dot") { pwd = rl::PairwiseDot(vec1, vec2, 0, vec1.size()); };
  BENCHMARK("Parallel Dot") { pd = rl::ParallelDot(vec1, vec2); };

  INFO("Dot     " << d << "\nPWD     " << pwd << "\nPar Dot " << pd);
  CHECK(std::abs(d - pwd) == Approx(0.f).margin(1.e-6f * sz));
  CHECK(std::abs(pd - pwd) == Approx(0.f).margin(1.e-6f * sz));
}
