#pragma once

#if __INTELLISENSE__
#undef __ARM_NEON
#undef __ARM_NEON__
#endif

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

namespace rl {

using FixZero = Eigen::type2index<0>; // Fix a dimension to zero in reshape/broadcast
using FixOne = Eigen::type2index<1>;  // Fix a dimension to one in reshape/broadcast

using B0 = Eigen::TensorFixedSize<bool, Eigen::Sizes<>>;
using B3 = Eigen::Tensor<bool, 3>;

using I0 = Eigen::TensorFixedSize<Index, Eigen::Sizes<>>;
using I1 = Eigen::Tensor<Index, 1>;
using I2 = Eigen::Tensor<Index, 2>;

using Re0 = Eigen::TensorFixedSize<float, Eigen::Sizes<>>; // Annoying return type for reductions
using Re1 = Eigen::Tensor<float, 1>;                       // 1D Real data
using Re2 = Eigen::Tensor<float, 2>;                       // 2D Real data
using Re3 = Eigen::Tensor<float, 3>;                       // 3D Real data
using Re4 = Eigen::Tensor<float, 4>;                       // 4D Real data
using Re5 = Eigen::Tensor<float, 5>;                       // 5D Real data

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

using Cxd1 = Eigen::Tensor<std::complex<double>, 1>; // 1D double precision complex data

// Useful shorthands
template<Index Rank>
using Sz = typename Eigen::DSizes<Index, Rank>;
using Sz1 = Sz<1>;
using Sz2 = Sz<2>;
using Sz3 = Sz<3>;
using Sz4 = Sz<4>;
using Sz5 = Sz<5>;
using Sz6 = Sz<6>;

template <typename T, int N, typename... Args>
decltype(auto) AddFront(Eigen::DSizes<T, N> const &back, Args... toAdd)
{
  std::array<Index, sizeof...(Args)> front = {{toAdd...}};
  Eigen::DSizes<Index, sizeof...(Args) + N> out;

  std::copy_n(front.begin(), sizeof...(Args), out.begin());
  std::copy_n(back.begin(), N, out.begin() + sizeof...(Args));
  return out;
}

template <typename T, int N, typename... Args>
decltype(auto) AddBack(Eigen::DSizes<T, N> const &front, Args... toAdd)
{
  std::array<Index, sizeof...(Args)> back = {{toAdd...}};
  Eigen::DSizes<Index, sizeof...(Args) + N> out;

  std::copy_n(front.begin(), N, out.begin());
  std::copy_n(back.begin(), sizeof...(Args), out.begin() + N);
  return out;
}

template <typename T, int N1, int N2>
decltype(auto) Concatenate(Eigen::DSizes<T, N1> const &a, Eigen::DSizes<T, N2> const &b)
{
  T constexpr Total = N1 + N2;
  Eigen::DSizes<T, Total> result;
  for (Index ii = 0; ii < N1; ii++) {
    result[ii] = a[ii];
  }
  for (Index ii = 0; ii < N2; ii++) {
    result[ii + N1] = b[ii];
  }
  return result;
}

template <size_t N, typename T>
Eigen::DSizes<typename T::value_type, N> FirstN(T const &sz)
{
  assert(N <= sz.size());
  Eigen::DSizes<typename T::value_type, N> first;
  std::copy_n(sz.begin(), N, first.begin());
  return first;
}

template <size_t N, typename T>
Eigen::DSizes<Index, N> LastN(T const &sz)
{
  assert(N <= sz.size());
  Eigen::DSizes<Index, N> last;
  std::copy_n(sz.end() - N, N, last.begin());
  return last;
}

template <size_t F, size_t N, typename T>
Eigen::DSizes<Index, N> MidN(T const &sz)
{
  assert(F + N <= sz.size());
  Eigen::DSizes<Index, N> out;
  std::copy_n(sz.begin() + F, N, out.begin());
  return out;
}

using Size3 = Eigen::Array<int16_t, 3, 1>;
using Point1 = Eigen::Matrix<float, 1, 1>;
using Point2 = Eigen::Matrix<float, 2, 1>;
using Point3 = Eigen::Matrix<float, 3, 1>;

template <typename T>
Index Product(T const &indices)
{
  return std::accumulate(indices.cbegin(), indices.cend(), 1, std::multiplies<Index>());
}

template <typename T>
T AMin(T const &a, T const &b) {
  T m;
  for (Index ii = 0; ii < a.size(); ii++) {
    m[ii] = std::min(a[ii], b[ii]);
  }
  return m;
}

} // namespace rl
