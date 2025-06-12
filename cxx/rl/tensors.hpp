#pragma once

#ifdef DEBUG
#define EIGEN_INITIALIZE_MATRICES_BY_NAN
#endif
// Need to define EIGEN_USE_THREADS before including these. This is done in CMakeLists.txt
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include "sys/threads.hpp"

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
  s.device(rl::Threads::TensorDevice()) = a.sum();
  return s();
}

template <typename T> typename T::Scalar Mean(T const &a)
{
  Eigen::TensorFixedSize<typename T::Scalar, Eigen::Sizes<>> s;
  s.device(rl::Threads::TensorDevice()) = a.mean();
  return s();
}

template <typename T> typename T::Scalar Minimum(T const &a)
{
  Eigen::TensorFixedSize<typename T::Scalar, Eigen::Sizes<>> m;
  m.device(rl::Threads::TensorDevice()) = a.minimum();
  return m();
}

template <typename T> auto NoNaNs(T const &a) -> T { return a.isfinite().select(a, a.constant(0.f)); }

template <typename T> typename T::Scalar Maximum(T const &a)
{
  Eigen::TensorFixedSize<typename T::Scalar, Eigen::Sizes<>> m;
  m.device(rl::Threads::TensorDevice()) = a.maximum();
  return m();
}

template <bool threads, typename T, typename U> inline decltype(auto) Dot(T const &a, U const &b)
{
  using Scalar = typename std::remove_reference<T>::type::Scalar;
  Eigen::TensorFixedSize<Scalar, Eigen::Sizes<>> d0;
  if constexpr (threads) {
    d0.device(rl::Threads::TensorDevice()) = (a * b.conjugate()).sum();
  } else {
    d0 = (a * b.conjugate()).sum();
  }
  Scalar const d = d0();
  return d;
}

template <bool threads, typename T> inline decltype(auto) Norm2(T const &a) { return std::real(Dot<threads>(a, a)); }
template <bool threads, typename T> inline decltype(auto) Norm(T const &a) { return std::sqrt(Norm2<threads>(a)); }

template <int D, typename T, typename U> inline decltype(auto) DimDot(T const &x, U const &y)
{
  Eigen::IndexList<Eigen::type2index<D>> dim;
  return (x * y.conjugate()).sum(dim);
}

template <typename Scalar, int N>
inline decltype(auto) AsTensorMap(Eigen::Vector<Scalar, Eigen::Dynamic> &x, Sz<N> const &shape)
{
  return Eigen::TensorMap<Eigen::Tensor<Scalar, N>>(x.data(), shape);
}

template <typename Scalar, int N>
inline decltype(auto) AsTensorMap(Eigen::Vector<Scalar, Eigen::Dynamic> const &x, Sz<N> const &shape)
{
  return Eigen::TensorMap<Eigen::Tensor<Scalar, N> const>(x.data(), shape);
}

template <typename Scalar, int N>
inline decltype(auto) AsConstTensorMap(Eigen::Array<Scalar, Eigen::Dynamic, 1> const &x, Sz<N> const &shape)
{
  return Eigen::TensorMap<Eigen::Tensor<Scalar, N> const>(x.data(), shape);
}

template <typename Scalar, int N>
inline decltype(auto) AsConstTensorMap(Eigen::Vector<Scalar, Eigen::Dynamic> &x, Sz<N> const &shape)
{
  return Eigen::TensorMap<Eigen::Tensor<Scalar, N> const>(x.data(), shape);
}

template <typename Scalar, int N>
inline decltype(auto) AsTensorMap(Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &x, Sz<N> const &shape)
{
  return Eigen::TensorMap<Eigen::Tensor<Scalar, N>>(x.data(), shape);
}

template <typename Scalar, int N>
inline decltype(auto) AsTensorMap(Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> const &x, Sz<N> const &shape)
{
  return Eigen::TensorMap<Eigen::Tensor<Scalar, N> const>(x.data(), shape);
}

template <typename T> inline decltype(auto) CollapseToArray(T &t)
{
  using Scalar = typename T::Scalar;
  typename Eigen::Array<Scalar, Eigen::Dynamic, 1>::AlignedMapType mapped(t.data(), t.size(), 1);
  return mapped;
}

template <typename T> inline decltype(auto) CollapseToArray(T const &t)
{
  using Scalar = typename T::Scalar;
  using Map = typename Eigen::Map<Eigen::Array<Scalar, Eigen::Dynamic, 1> const, Eigen::AlignedMax>;
  Map mapped(t.data(), t.size(), 1);
  return mapped;
}

template <typename T> inline decltype(auto) CollapseToVector(T &t)
{
  using Scalar = typename T::Scalar;
  typename Eigen::Matrix<Scalar, Eigen::Dynamic, 1>::AlignedMapType mapped(t.data(), t.size(), 1);
  return mapped;
}

template <typename T> inline decltype(auto) CollapseToConstVector(T &t)
{
  using Scalar = typename T::Scalar;
  using Map = typename Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, 1> const, Eigen::AlignedMax>;
  Map mapped(t.data(), t.size(), 1);
  return mapped;
}

template <typename T> inline decltype(auto) CollapseToVector(T const &t)
{
  using Scalar = typename T::Scalar;
  using Map = typename Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, 1> const, Eigen::AlignedMax>;
  Map mapped(t.data(), t.size(), 1);
  return mapped;
}

template <typename T, int toCollapse = 1> inline decltype(auto) CollapseToMatrix(T &t)
{
  using Scalar = typename T::Scalar;
  Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> mapped(
    t.data(), std::accumulate(t.dimensions().begin(), t.dimensions().begin() + toCollapse, 1L, std::multiplies<Eigen::Index>()),
    std::accumulate(t.dimensions().begin() + toCollapse, t.dimensions().end(), 1L, std::multiplies<Eigen::Index>()));
  return mapped;
}

template <typename T, int toCollapse = 1> inline decltype(auto) CollapseToMatrix(T const &t)
{
  using Scalar = typename T::Scalar;
  Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> const> mapped(
    t.data(), std::accumulate(t.dimensions().begin(), t.dimensions().begin() + toCollapse, 1L, std::multiplies<Eigen::Index>()),
    std::accumulate(t.dimensions().begin() + toCollapse, t.dimensions().end(), 1L, std::multiplies<Eigen::Index>()));
  return mapped;
}

template <typename T, int toCollapse = 1> inline decltype(auto) CollapseToConstMatrix(T const &t)
{
  using Scalar = typename T::Scalar;
  Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> const> mapped(
    t.data(), std::accumulate(t.dimensions().begin(), t.dimensions().begin() + toCollapse, 1L, std::multiplies<Eigen::Index>()),
    std::accumulate(t.dimensions().begin() + toCollapse, t.dimensions().end(), 1L, std::multiplies<Eigen::Index>()));
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

template <typename T> inline auto ConstMap(Eigen::TensorMap<T> &x) -> Eigen::TensorMap<T const>
{
  Eigen::TensorMap<T const> cx(x.data(), x.dimensions());
  return cx;
}

template <typename T> inline auto ConstMap(T &x) -> Eigen::TensorMap<T const>
{
  Eigen::TensorMap<T const> cx(x.data(), x.dimensions());
  return cx;
}

template <typename T> inline auto Map(T &x) -> Eigen::TensorMap<T>
{
  Eigen::TensorMap<T> xm(x.data(), x.dimensions());
  return xm;
}

} // namespace rl
