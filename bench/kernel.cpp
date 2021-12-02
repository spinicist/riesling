#define CATCH_CONFIG_ENABLE_BENCHMARKING

#include <Eigen/Dense>
#include <catch2/catch.hpp>
#include <unsupported/Eigen/CXX11/Tensor>

using Point3 = Eigen::Matrix<float, 3, 1>;

template <int IP, int TP>
Eigen::TensorFixedSize<float, Eigen::Sizes<IP, IP, TP>> DistSq(Point3 const p)
{
  using KTensor = Eigen::TensorFixedSize<float, Eigen::Sizes<IP, IP, TP>>;
  using KArray = Eigen::TensorFixedSize<float, Eigen::Sizes<IP>>;
  using FixOne = Eigen::type2index<1>;
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
    auto const kx = (indices.constant(p[0]) - indices).square().reshape(rshX).broadcast(brdX);
    auto const ky = (indices.constant(p[1]) - indices).square().reshape(rshY).broadcast(brdY);
    auto const kz = (indices.constant(p[2]) - indices).square().reshape(rshZ).broadcast(brdZ);
    k = kx + ky + kz;
  } else {
    constexpr Eigen::IndexList<FixIn, FixOne, FixOne> rshX;
    constexpr Eigen::IndexList<FixOne, FixIn, FixOne> brdX;
    constexpr Eigen::IndexList<FixOne, FixIn, FixOne> rshY;
    constexpr Eigen::IndexList<FixIn, FixOne, FixOne> brdY;
    auto const kx = (indices.constant(p[0]) - indices).square().reshape(rshX).broadcast(brdX);
    auto const ky = (indices.constant(p[1]) - indices).square().reshape(rshY).broadcast(brdY);
    k = kx + ky;
  }
  return k;
}

TEST_CASE("DistSq")
{
  auto const z = Point3::Zero();

  BENCHMARK("3-1")
  {
    DistSq<3, 1>(z);
  };

  BENCHMARK("3-3")
  {
    DistSq<3, 3>(z);
  };

  BENCHMARK("5-1")
  {
    DistSq<5, 1>(z);
  };

  BENCHMARK("5-5")
  {
    DistSq<5, 5>(z);
  };
}