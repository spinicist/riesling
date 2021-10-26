#define CATCH_CONFIG_ENABLE_BENCHMARKING

#include <catch2/catch.hpp>
#include <unsupported/Eigen/CXX11/Tensor>

TEST_CASE("Dot")
{
  using Cx4 = Eigen::Tensor<std::complex<float>, 4>;
  int const sz = 256;
  Cx4 grid(sz, sz, sz, 16);
  grid.setRandom();
  Eigen::ThreadPool gp(std::thread::hardware_concurrency());
  Eigen::ThreadPoolDevice dev(&gp, gp.NumThreads());

  BENCHMARK("Naive")
  {
    Eigen::TensorFixedSize<std::complex<float>, Eigen::Sizes<>> d;
    d = (grid.conjugate() * grid).sum();
  };

  BENCHMARK("Threaded")
  {
    Eigen::TensorFixedSize<std::complex<float>, Eigen::Sizes<>> d;
    d.device(dev) = (grid.conjugate() * grid).sum();
  };
}