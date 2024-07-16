#pragma once

// Intellisense gives false positives with Eigen+ARM
#if __INTELLISENSE__
#undef __ARM_NEON
#undef __ARM_NEON__
#endif

// This doesn't actually help with complex matrices as std::complex has no NaN
#ifdef DEBUG
#define EIGEN_INITIALIZE_MATRICES_BY_NAN
#endif

// Need to define EIGEN_USE_THREADS before including these. This is done in CMakeLists.txt
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include <cassert>
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
template <int N> using ReN = Eigen::Tensor<float, N>;
using Re1 = ReN<1>;
using Re2 = ReN<2>;
using Re3 = ReN<3>;
using Re4 = ReN<4>;
using Re5 = ReN<5>;

using Rd1 = Eigen::Tensor<double, 1>;
using Rd4 = Eigen::Tensor<double, 4>;

using Cx = std::complex<float>;
using Cxd = std::complex<double>;

using Cx0 = Eigen::TensorFixedSize<Cx, Eigen::Sizes<>>;
template <int N> using CxN = Eigen::Tensor<Cx, N>;
using Cx1 = CxN<1>;
using Cx2 = CxN<2>;
using Cx3 = CxN<3>;
using Cx4 = CxN<4>;
using Cx5 = CxN<5>;
using Cx6 = CxN<6>;
using Cx7 = CxN<7>;

template <int N> using CxNMap = Eigen::TensorMap<CxN<N>>;
using Cx1Map = CxNMap<1>;
using Cx2Map = CxNMap<2>;
using Cx3Map = CxNMap<3>;
using Cx4Map = CxNMap<4>;
using Cx5Map = CxNMap<5>;
using Cx6Map = CxNMap<6>;
using Cx7Map = CxNMap<7>;

template <int N> using CxNCMap = Eigen::TensorMap<CxN<N> const>;
using Cx1CMap = CxNCMap<1>;
using Cx2CMap = CxNCMap<2>;
using Cx3CMap = CxNCMap<3>;
using Cx4CMap = CxNCMap<4>;
using Cx5CMap = CxNCMap<5>;
using Cx6CMap = CxNCMap<6>;
using Cx7CMap = CxNCMap<7>;

using Cxd1 = Eigen::Tensor<std::complex<double>, 1>; // 1D double precision complex data

// Useful shorthands
template <int Rank> using Sz = typename Eigen::DSizes<Index, Rank>;
using Sz1 = Sz<1>;
using Sz2 = Sz<2>;
using Sz3 = Sz<3>;
using Sz4 = Sz<4>;
using Sz5 = Sz<5>;
using Sz6 = Sz<6>;
using Sz7 = Sz<7>;

using Size3 = Eigen::Array<int16_t, 3, 1>;
using Point1 = Eigen::Matrix<float, 1, 1>;
using Point2 = Eigen::Matrix<float, 2, 1>;
using Point3 = Eigen::Matrix<float, 3, 1>;

template<int N> auto Constant(Index const c) -> Sz<N> {
  Sz<N> C;
  C.fill(c);
  return C;
}

template <typename T, int N, typename... Args> decltype(auto) AddFront(Eigen::DSizes<T, N> const &back, Args... toAdd)
{
  static_assert(sizeof...(Args) > 0);
  Eigen::DSizes<T, sizeof...(Args)>     front{{toAdd...}};
  Eigen::DSizes<T, sizeof...(Args) + N> out;

  std::copy_n(front.begin(), sizeof...(Args), out.begin());
  std::copy_n(back.begin(), N, out.begin() + sizeof...(Args));
  return out;
}

template <typename T, int N, typename... Args> decltype(auto) AddBack(Eigen::DSizes<T, N> const &front, Args... toAdd)
{
  static_assert(sizeof...(Args) > 0);
  Eigen::DSizes<T, sizeof...(Args)>     back{{toAdd...}};
  Eigen::DSizes<T, sizeof...(Args) + N> out;

  std::copy_n(front.begin(), N, out.begin());
  std::copy_n(back.begin(), sizeof...(Args), out.begin() + N);
  return out;
}

template <typename T, int N1, int N2> decltype(auto) Concatenate(Eigen::DSizes<T, N1> const &a, Eigen::DSizes<T, N2> const &b)
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

template <size_t N, typename T> auto FirstN(T const &sz) -> Eigen::DSizes<typename T::value_type, N>
{
  assert(N <= sz.size());
  Eigen::DSizes<typename T::value_type, N> first;
  std::copy_n(sz.begin(), N, first.begin());
  return first;
}

template <size_t N, typename T> auto LastN(T const &sz) -> Eigen::DSizes<typename T::value_type, N>
{
  assert(N <= sz.size());
  Eigen::DSizes<Index, N> last;
  std::copy_n(sz.end() - N, N, last.begin());
  return last;
}

template <size_t F, size_t N, typename T> auto MidN(T const &sz) -> Eigen::DSizes<typename T::value_type, N>
{
  assert(F + N <= sz.size());
  Eigen::DSizes<Index, N> out;
  std::copy_n(sz.begin() + F, N, out.begin());
  return out;
}

template <int N> Index Product(Sz<N> const &indices)
{
  return std::accumulate(indices.cbegin(), indices.cend(), 1L, std::multiplies<Index>());
}

template <typename T> T AMin(T const &a, T const &b)
{
  T m;
  for (size_t ii = 0; ii < a.size(); ii++) {
    m[ii] = std::min(a[ii], b[ii]);
  }
  return m;
}

template <int N> auto Add(Eigen::DSizes<Index, N> const &sz, Index const a) -> Eigen::DSizes<Index, N>
{
  Eigen::DSizes<Index, N> result;
  std::transform(sz.begin(), sz.begin() + N, result.begin(), [a](Index const i) { return i + a; });
  return result;
}

template <int N, typename T> auto Mul(Eigen::DSizes<Index, N> const &sz, T const m) -> Eigen::DSizes<Index, N>
{
  Eigen::DSizes<Index, N> result;
  std::transform(sz.begin(), sz.begin() + N, result.begin(), [m](Index const i) { return i * m; });
  return result;
}

template <int N, typename T> auto Div(Eigen::DSizes<Index, N> const &sz, T const d) -> Eigen::DSizes<Index, N>
{
  Eigen::DSizes<Index, N> result;
  std::transform(sz.begin(), sz.begin() + N, result.begin(), [d](Index const i) { return i / d; });
  return result;
}

template <typename T> auto Wrap(T const index, T const sz) -> T
{
  T const t = index + sz;
  T const w = t - sz * (t / sz);
  return w;
}

} // namespace rl
