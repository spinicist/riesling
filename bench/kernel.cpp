#define CATCH_CONFIG_ENABLE_BENCHMARKING

#include <Eigen/Dense>
#include <catch2/catch.hpp>
#include <unsupported/Eigen/CXX11/Tensor>

#include "../src/kernel.hpp"

using Point3 = Eigen::Matrix<float, 3, 1>;

using namespace rl;

template <int IP, int TP>
Eigen::TensorFixedSize<float, Eigen::Sizes<IP, IP, TP>> DistSq(Point3 const p)
{
  using KTensor = Eigen::TensorFixedSize<float, Eigen::Sizes<IP, IP, TP>>;
  using KArray = Eigen::TensorFixedSize<float, Eigen::Sizes<IP>>;
  using FixIn = Eigen::type2index<IP>;
  KArray indices;
  std::iota(indices.data(), indices.data() + IP, -IP / 2); // Note INTEGER division
  KTensor k;
  if constexpr (TP > 1) {
    constexpr Eigen::IndexList<FixIn, FixOne, FixOne> rshX;
    constexpr Eigen::IndexList<FixOne, FixIn, FixIn> brdX;
    constexpr Eigen::IndexList<FixOne, FixIn, FixOne> rshY;
    constexpr Eigen::IndexList<FixIn, FixOne, FixIn> brdY;
    constexpr Eigen::IndexList<FixOne, FixOne, FixIn> rshZ;
    constexpr Eigen::IndexList<FixIn, FixIn, FixOne> brdZ;
    auto const kx = ((indices.constant(p[0]) - indices) / indices.constant(IP / 2.f))
                      .square()
                      .reshape(rshX)
                      .broadcast(brdX);
    auto const ky = ((indices.constant(p[1]) - indices) / indices.constant(IP / 2.f))
                      .square()
                      .reshape(rshY)
                      .broadcast(brdY);
    auto const kz = ((indices.constant(p[2]) - indices) / indices.constant(TP / 2.f))
                      .square()
                      .reshape(rshZ)
                      .broadcast(brdZ);
    k = kx + ky + kz;
  } else {
    constexpr Eigen::IndexList<FixIn, FixOne, FixOne> rshX;
    constexpr Eigen::IndexList<FixOne, FixIn, FixOne> brdX;
    constexpr Eigen::IndexList<FixOne, FixIn, FixOne> rshY;
    constexpr Eigen::IndexList<FixIn, FixOne, FixOne> brdY;
    auto const kx = ((indices.constant(p[0]) - indices) / indices.constant(IP / 2.f))
                      .square()
                      .reshape(rshX)
                      .broadcast(brdX);
    auto const ky = ((indices.constant(p[1]) - indices) / indices.constant(IP / 2.f))
                      .square()
                      .reshape(rshY)
                      .broadcast(brdY);
    k = kx + ky;
  }
  return k;
}

template <int IP, int TP>
Eigen::TensorFixedSize<float, Eigen::Sizes<IP, IP, TP>> Naive(Point3 const p)
{
  using KTensor = Eigen::TensorFixedSize<float, Eigen::Sizes<IP, IP, TP>>;
  KTensor k;
  for (Index iz = 0; iz < TP; iz++) {
    for (Index iy = 0; iy < IP; iy++) {
      for (Index ix = 0; ix < IP; ix++) {
        k(ix, iy, iz) = ((p - Point3(ix, iy, iz)) / (IP / 2.f)).squaredNorm();
      }
    }
  }
  return k;
}

TEST_CASE("Kernels")
{
  auto const z = Point3::Zero();
  KaiserBessel<3, 3> kb(2.f);
  FlatIron<3, 3> fi(2.f);

  BENCHMARK("Old")
  {
    DistSq<3, 3>(z);
  };

  BENCHMARK("Current")
  {
    kb.distSq(z);
  };

  BENCHMARK("Naive")
  {
    Naive<3, 3>(z);
  };

  BENCHMARK("KB")
  {
    kb.k(z);
  };

  BENCHMARK("FI")
  {
    fi.k(z);
  };
}