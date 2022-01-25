#pragma once

#ifdef DEBUG
#define EIGEN_INITIALIZE_MATRICES_BY_NAN
#endif
// Need to define EIGEN_USE_THREADS before including these. This is done in CMakeLists.txt
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include <complex>

using Index = Eigen::Index;

namespace Eigen {
using Array3l = Array<Index, 3, 1>;
using ArrayXl = Array<Index, -1, 1>;
} // namespace Eigen

using FixZero = Eigen::type2index<0>; // Fix a dimension to zero in reshape/broadcast
using FixOne = Eigen::type2index<1>;  // Fix a dimension to one in reshape/broadcast

using B0 = Eigen::TensorFixedSize<bool, Eigen::Sizes<>>;
using B3 = Eigen::Tensor<bool, 3>;

using I0 = Eigen::TensorFixedSize<Index, Eigen::Sizes<>>;
using I1 = Eigen::Tensor<Index, 1>;
using I2 = Eigen::Tensor<Index, 2>;

using R0 = Eigen::TensorFixedSize<float, Eigen::Sizes<>>; // Annoying return type for reductions
using R1 = Eigen::Tensor<float, 1>;                       // 1D Real data
using R2 = Eigen::Tensor<float, 2>;                       // 2D Real data
using R3 = Eigen::Tensor<float, 3>;                       // 3D Real data
using R4 = Eigen::Tensor<float, 4>;                       // 4D Real data
using R5 = Eigen::Tensor<float, 5>;                       // 5D Real data

using Rd1 = Eigen::Tensor<double, 1>;

using Cx = std::complex<float>;
using Cxd = std::complex<double>;

using Cx0 = Eigen::TensorFixedSize<Cx, Eigen::Sizes<>>;
using Cx1 = Eigen::Tensor<Cx, 1>; // 1D Complex data
using Cx2 = Eigen::Tensor<Cx, 2>; // 2D Complex data
using Cx3 = Eigen::Tensor<Cx, 3>; // 3D Complex data
using Cx4 = Eigen::Tensor<Cx, 4>; // 4D Complex data...spotted a pattern yet?
using Cx5 = Eigen::Tensor<Cx, 5>;
using Cx6 = Eigen::Tensor<Cx, 6>;
using Cx7 = Eigen::Tensor<Cx, 7>;

using Cxd1 = Eigen::Tensor<std::complex<double>, 1>; // 1D double precision complex data

// Useful shorthands
using Sz1 = Cx1::Dimensions;
using Sz2 = Cx2::Dimensions;
using Sz3 = Cx3::Dimensions;
using Sz4 = Cx4::Dimensions;
using Sz5 = Cx5::Dimensions;

template <typename T, int N>
Eigen::DSizes<T, N + 1> AddFront(Eigen::DSizes<T, N> const &sz, Index const ii)
{
  Eigen::DSizes<Index, N + 1> out;
  out[0] = ii;
  std::copy_n(sz.begin(), N, out.begin() + 1);
  return out;
}

template <typename T, int N>
Eigen::DSizes<T, N + 1> AddBack(Eigen::DSizes<T, N> const &sz, Index const ii)
{
  Eigen::DSizes<Index, N + 1> out;
  std::copy_n(sz.begin(), N, out.begin());
  out[N] = ii;
  return out;
}

template <int N, typename T>
Eigen::DSizes<Index, N> LastN(T const &sz)
{
  assert(N <= sz.size());
  Eigen::DSizes<Index, N> last;
  std::copy_n(sz.end() - N, N, last.begin());
  return last;
}

using Size3 = Eigen::Array<int16_t, 3, 1>;
using Point3 = Eigen::Matrix<float, 3, 1>;
