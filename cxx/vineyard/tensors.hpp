#pragma once

#ifdef DEBUG
#define EIGEN_INITIALIZE_MATRICES_BY_NAN
#endif
// Need to define EIGEN_USE_THREADS before including these. This is done in CMakeLists.txt
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include "threads.hpp"

namespace rl {

// Tensor operations
template <typename T> decltype(auto) Transpose(T const &a)
{
  assert(a.NumDimensions == 1);
  return a.reshape(Eigen::array<Index, 2>{1, a.size()});
}

template <typename T> typename T::Scalar Sum(T const &a)
{
  Eigen::TensorFixedSize<typename T::Scalar, Eigen::Sizes<>> s;
  s.device(rl::Threads::GlobalDevice()) = a.sum();
  return s();
}

template <typename T> typename T::Scalar Mean(T const &a)
{
  Eigen::TensorFixedSize<typename T::Scalar, Eigen::Sizes<>> s;
  s.device(rl::Threads::GlobalDevice()) = a.mean();
  return s();
}

template <typename T> typename T::Scalar Minimum(T const &a)
{
  Eigen::TensorFixedSize<typename T::Scalar, Eigen::Sizes<>> m;
  m.device(rl::Threads::GlobalDevice()) = a.minimum();
  return m();
}

template <typename T> auto NoNaNs(T const &a) -> T { return a.isfinite().select(a, a.constant(0.f)); }

template <typename T> typename T::Scalar Maximum(T const &a)
{
  Eigen::TensorFixedSize<typename T::Scalar, Eigen::Sizes<>> m;
  m.device(rl::Threads::GlobalDevice()) = a.maximum();
  return m();
}

template <typename T1, typename T2> typename T1::Scalar Dot(T1 const &a, T2 const &b)
{
  Eigen::TensorFixedSize<typename T1::Scalar, Eigen::Sizes<>> d;
  d.device(rl::Threads::GlobalDevice()) = (a.conjugate() * b).sum();
  return d();
}

template <typename T> float Norm2(T const &a) { return std::real(Dot(a, a)); }

template <typename T> float Norm(T const &a) { return std::sqrt(std::real(Dot(a, a))); }

template <typename T, typename U> inline decltype(auto) ConjugateSum(T &&x, U &&y)
{
  Eigen::IndexList<Eigen::type2index<0>> zero;
  return (x * y.conjugate()).sum(zero);
}

template <typename T> inline decltype(auto) FirstToLast4(T const &x)
{
  Eigen::IndexList<Eigen::type2index<1>, Eigen::type2index<2>, Eigen::type2index<3>, Eigen::type2index<0>> indices;
  return x.shuffle(indices);
}

template <typename T1, typename T2, int D = 0> inline decltype(auto) Contract(T1 const &a, T2 const &b)
{
  return a.contract(b, Eigen::IndexPairList<Eigen::type2indexpair<D, D>>());
}

template <typename T1, typename T2> inline decltype(auto) Outer(T1 const &a, T2 const &b)
{
  constexpr Eigen::array<Eigen::IndexPair<Index>, 0> empty = {};
  return a.contract(b, empty);
}

template <typename T> inline decltype(auto) ExpandSum(T const &a, T const &b, T const &c)
{
  using FixedOne = Eigen::type2index<1>;
  Eigen::IndexList<int, FixedOne, FixedOne> rsh_a;
  Eigen::IndexList<FixedOne, int, int>      brd_a;
  rsh_a.set(0, a.size());
  brd_a.set(1, b.size());
  brd_a.set(2, c.size());
  Eigen::IndexList<FixedOne, int, FixedOne> rsh_b;
  Eigen::IndexList<int, FixedOne, int>      brd_b;
  brd_b.set(0, a.size());
  rsh_b.set(1, b.size());
  brd_b.set(2, c.size());
  Eigen::IndexList<FixedOne, FixedOne, int> rsh_c;
  Eigen::IndexList<int, int, FixedOne>      brd_c;
  brd_c.set(0, a.size());
  brd_c.set(1, b.size());
  rsh_c.set(2, c.size());
  return a.reshape(rsh_a).broadcast(brd_a) + b.reshape(rsh_b).broadcast(brd_b) + c.reshape(rsh_c).broadcast(brd_c);
}

template <typename Scalar, int N> inline decltype(auto) Tensorfy(Eigen::Vector<Scalar, Eigen::Dynamic> &x, Sz<N> const &shape)
{
  return Eigen::TensorMap<Eigen::Tensor<Scalar, N>>(x.data(), shape);
}

template <typename Scalar, int N>
inline decltype(auto) Tensorfy(Eigen::Vector<Scalar, Eigen::Dynamic> const &x, Sz<N> const &shape)
{
  return Eigen::TensorMap<Eigen::Tensor<Scalar, N> const>(x.data(), shape);
}

template <typename Scalar, int N>
inline decltype(auto) Tensorfy(Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &x, Sz<N> const &shape)
{
  return Eigen::TensorMap<Eigen::Tensor<Scalar, N>>(x.data(), shape);
}

template <typename Scalar, int N>
inline decltype(auto) Tensorfy(Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> const &x, Sz<N> const &shape)
{
  return Eigen::TensorMap<Eigen::Tensor<Scalar, N> const>(x.data(), shape);
}

template <typename T> inline decltype(auto) CollapseToArray(T &t)
{
  using Scalar = typename T::Scalar;
  typename Eigen::Array<Scalar, Eigen::Dynamic, 1>::AlignedMapType mapped(
    t.data(), std::accumulate(t.dimensions().begin(), t.dimensions().end(), 1, std::multiplies<Eigen::Index>()), 1);
  return mapped;
}

template <typename T> inline decltype(auto) CollapseToArray(T const &t)
{
  using Scalar = typename T::Scalar;
  typename Eigen::Array<Scalar, Eigen::Dynamic, 1>::ConstAlignedMapType mapped(
    t.data(), std::accumulate(t.dimensions().begin(), t.dimensions().end(), 1, std::multiplies<Eigen::Index>()), 1);
  return mapped;
}

template <typename T> inline decltype(auto) CollapseToVector(T &t)
{
  using Scalar = typename T::Scalar;
  typename Eigen::Matrix<Scalar, Eigen::Dynamic, 1>::AlignedMapType mapped(
    t.data(), std::accumulate(t.dimensions().begin(), t.dimensions().end(), 1, std::multiplies<Eigen::Index>()), 1);
  return mapped;
}

template <typename T> inline decltype(auto) CollapseToConstVector(T &t)
{
  using Scalar = typename T::Scalar;
  typename Eigen::Matrix<Scalar, Eigen::Dynamic, 1>::ConstAlignedMapType mapped(
    t.data(), std::accumulate(t.dimensions().begin(), t.dimensions().end(), 1, std::multiplies<Eigen::Index>()), 1);
  return mapped;
}

template <typename T> inline decltype(auto) CollapseToVector(T const &t)
{
  using Scalar = typename T::Scalar;
  typename Eigen::Matrix<Scalar, Eigen::Dynamic, 1>::ConstAlignedMapType mapped(
    t.data(), std::accumulate(t.dimensions().begin(), t.dimensions().end(), 1, std::multiplies<Eigen::Index>()), 1);
  return mapped;
}

template <typename T, int toCollapse = 1> inline decltype(auto) CollapseToMatrix(T &t)
{
  using Scalar = typename T::Scalar;
  Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> mapped(
    t.data(), std::accumulate(t.dimensions().begin(), t.dimensions().begin() + toCollapse, 1, std::multiplies<Eigen::Index>()),
    std::accumulate(t.dimensions().begin() + toCollapse, t.dimensions().end(), 1, std::multiplies<Eigen::Index>()));
  return mapped;
}

template <typename T, int toCollapse = 1> inline decltype(auto) CollapseToMatrix(T const &t)
{
  using Scalar = typename T::Scalar;
  Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> const> mapped(
    t.data(), std::accumulate(t.dimensions().begin(), t.dimensions().begin() + toCollapse, 1, std::multiplies<Eigen::Index>()),
    std::accumulate(t.dimensions().begin() + toCollapse, t.dimensions().end(), 1, std::multiplies<Eigen::Index>()));
  return mapped;
}

template <typename T, int toCollapse = 1> inline decltype(auto) CollapseToConstMatrix(T const &t)
{
  using Scalar = typename T::Scalar;
  Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> const> mapped(
    t.data(), std::accumulate(t.dimensions().begin(), t.dimensions().begin() + toCollapse, 1, std::multiplies<Eigen::Index>()),
    std::accumulate(t.dimensions().begin() + toCollapse, t.dimensions().end(), 1, std::multiplies<Eigen::Index>()));
  return mapped;
}

template <typename T> inline auto ChipMap(T &a, Index const index)
{
  constexpr auto LastDim = T::NumDimensions - 1;
  using Scalar = typename T::Scalar;
  using Tensor = Eigen::Tensor<Scalar, LastDim>;
  assert(index < a.dimension(LastDim));
  auto const chipDims = FirstN<LastDim>(a.dimensions());
  return Eigen::TensorMap<Tensor>(a.data() + Product(chipDims) * index, chipDims);
}

template <typename T> inline auto CChipMap(T const &a, Index const index)
{
  constexpr auto LastDim = T::NumDimensions - 1;
  using Scalar = typename T::Scalar;
  using Tensor = Eigen::Tensor<Scalar, LastDim>;
  assert(index < a.dimension(LastDim));
  auto const chipDims = FirstN<LastDim>(a.dimensions());
  return Eigen::TensorMap<Tensor const>(a.data() + Product(chipDims) * index, chipDims);
}

template <typename T> inline auto ConstMap(Eigen::TensorMap<T> x)
{
  Eigen::TensorMap<T const> cx(x.data(), x.dimensions());
  return cx;
}

template <typename T> inline auto ConstMap(T x)
{
  Eigen::TensorMap<T const> cx(x.data(), x.dimensions());
  return cx;
}

template <typename T> inline auto Map(T x) { return Eigen::TensorMap<T>(x.data(), x.dimensions()); }

} // namespace rl
