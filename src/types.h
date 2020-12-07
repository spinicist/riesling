#pragma once

#ifdef DEBUG
#define EIGEN_INITIALIZE_MATRICES_BY_NAN
#endif
// Need to define EIGEN_USE_THREADS before including these. This is done in CMakeLists.txt
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include <complex>

using Array3l = Eigen::Array<long, 3, 1>;
using ArrayXl = Eigen::Array<long, -1, 1>;

using B0 = Eigen::TensorFixedSize<bool, Eigen::Sizes<>>;

using L0 = Eigen::TensorFixedSize<long, Eigen::Sizes<>>;
using L1 = Eigen::Tensor<long, 1>;
using L2 = Eigen::Tensor<long, 2>;

using R0 = Eigen::TensorFixedSize<float, Eigen::Sizes<>>; // Annoying return type for reductions
using R1 = Eigen::Tensor<float, 1>;                       // 1D Real data
using R2 = Eigen::Tensor<float, 2>;                       // 2D Real data
using R3 = Eigen::Tensor<float, 3>;                       // 3D Real data
using R4 = Eigen::Tensor<float, 4>;                       // 4D Real data

using Cx0 = Eigen::TensorFixedSize<std::complex<float>, Eigen::Sizes<>>;
using Cx1 = Eigen::Tensor<std::complex<float>, 1>; // 1D Complex data
using Cx2 = Eigen::Tensor<std::complex<float>, 2>; // 2D Complex data
using Cx3 = Eigen::Tensor<std::complex<float>, 3>; // 3D Complex data
using Cx4 = Eigen::Tensor<std::complex<float>, 4>; // 4D Complex data...spotted a pattern yet?

// Useful shorthands
using Sz1 = Eigen::array<long, 1>;
using Sz2 = Eigen::array<long, 2>;
using Sz3 = Eigen::array<long, 3>;
using Sz4 = Eigen::array<long, 4>;
using Dims2 = Cx2::Dimensions;
using Dims3 = Cx3::Dimensions;
using Dims4 = Cx4::Dimensions;
using Size2 = Eigen::Array<Eigen::Index, 2, 1>;
using Size3 = Eigen::Array<Eigen::Index, 3, 1>;
using Point3 = Eigen::Array<float, 3, 1>;
using Point4 = Eigen::Array<float, 4, 1>;
using Points3 = Eigen::Array<float, 3, -1>;
using Points4 = Eigen::Array<float, 4, -1>;
using Pads3 = Eigen::array<std::pair<long, long>, 3>;
using Size4 = Eigen::Array<Eigen::Index, 4, 1>;

// This is the type of the lambda functions to represent the encode/decode operators
using EncodeFunction = std::function<void(Cx3 &cartesian, Cx3 &radial)>;
using DecodeFunction = std::function<void(Cx3 const &radial, Cx3 &cartesian)>;
using SystemFunction = std::function<void(Cx3 const &A, Cx3 &B)>;

// Tensor operations
template <typename T>
float dot(T const &a, T const &b)
{
  R0 d = (a * b.conjugate()).real().sum();
  return d();
}

template <typename T>
float norm2(T const &a)
{
  return dot(a, a);
}

template <typename T>
float norm(T const &a)
{
  return sqrt(dot(a, a));
}

template <typename T>
inline decltype(auto) wrap(T const &index, long const &sz)
{
  auto const t = index + sz;
  auto const w = t - sz * (t / sz);
  return w;
}

template <typename T1, typename T2>
inline decltype(auto) wrap(T1 const &index, T2 const &sz)
{
  auto const t = index + sz;
  auto const w = t - sz * (t / sz);
  return w;
}

template <typename T>
inline decltype(auto) tile(T &&x, long const N)
{
  return x.reshape(Sz4{1, x.dimension(0), x.dimension(1), x.dimension(2)})
      .broadcast(Sz4{N, 1, 1, 1});
}
