#pragma once

#ifdef DEBUG
#define EIGEN_INITIALIZE_MATRICES_BY_NAN
#endif
// Need to define EIGEN_USE_THREADS before including these. This is done in CMakeLists.txt
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

// Tensor operations
template <typename T>
decltype(auto) Transpose(T const &a)
{
  assert(a.NumDimensions == 1);
  return a.reshape(Eigen::array<long, 2>{1, a.size()});
}

template <typename T>
typename T::Scalar Sum(T const &a)
{
  Eigen::TensorFixedSize<typename T::Scalar, Eigen::Sizes<>> s = a.sum();
  return s();
}

template <typename T>
typename T::Scalar Maximum(T const &a)
{
  Eigen::TensorFixedSize<typename T::Scalar, Eigen::Sizes<>> m = a.maximum();
  return m();
}

template <typename T1, typename T2>
typename T1::Scalar Dot(T1 const &a, T2 const &b)
{
  Eigen::TensorFixedSize<typename T1::Scalar, Eigen::Sizes<>> d = (a.conjugate() * b).sum();
  return d();
}

template <typename T>
float Norm2(T const &a)
{
  return std::real(Dot(a, a));
}

template <typename T>
float Norm(T const &a)
{
  return sqrt(std::real(Dot(a, a)));
}

template <typename T>
inline decltype(auto) Wrap(T const &index, long const &sz)
{
  auto const t = index + sz;
  auto const w = t - sz * (t / sz);
  return w;
}

template <typename T1, typename T2>
inline decltype(auto) Wrap(T1 const &index, T2 const &sz)
{
  auto const t = index + sz;
  auto const w = t - sz * (t / sz);
  return w;
}

template <typename T, typename U>
inline decltype(auto) ConjugateSum(T &&x, U &&y)
{
  Eigen::IndexList<Eigen::type2index<0>> zero;
  return (x * y.conjugate()).sum(zero);
}

template <typename T>
inline decltype(auto) FirstToLast4(T const &x)
{
  Eigen::IndexList<Eigen::type2index<1>,
                   Eigen::type2index<2>,
                   Eigen::type2index<3>,
                   Eigen::type2index<0>> indices;
  return x.shuffle(indices);
}

template <typename T>
inline decltype(auto) Tile(T &&x, long const N)
{
  Eigen::IndexList<Eigen::type2index<1>, int, int, int> res;
  res.set(1, x.dimension(0));
  res.set(2, x.dimension(1));
  res.set(3, x.dimension(2));
  Eigen::IndexList<int, Eigen::type2index<1>, Eigen::type2index<1>, Eigen::type2index<1>> brd;
  brd.set(0, N);
  return x.reshape(res).broadcast(brd);
}

template <typename T, typename U>
inline decltype(auto) TileToMatch(T &&x, U const &dims)
{
  using FixedOne = Eigen::type2index<1>;
  Eigen::IndexList<FixedOne, int, int, int> res;
  res.set(1, dims[1]);
  res.set(2, dims[2]);
  res.set(3, dims[3]);
  Eigen::IndexList<int, FixedOne, FixedOne, FixedOne> brd;
  brd.set(0, dims[0]);
  return x.reshape(res).broadcast(brd);
}

template <typename T1, typename T2, int D = 0>
inline decltype(auto) Contract(T1 const &a, T2 const &b)
{
  return a.contract(b, Eigen::IndexPairList<Eigen::type2indexpair<D, D>>());
}

template <typename T1, typename T2>
inline decltype(auto) Outer(T1 const &a, T2 const &b)
{
  constexpr Eigen::array<Eigen::IndexPair<long>, 0> empty = {};
  return a.contract(b, empty);
}

template <typename T>
inline decltype(auto) ExpandSum(T const &a, T const &b, T const &c)
{
  using FixedOne = Eigen::type2index<1>;
  Eigen::IndexList<int, FixedOne, FixedOne> rsh_a;
  Eigen::IndexList<FixedOne, int, int> brd_a;
  rsh_a.set(0, a.size());
  brd_a.set(1, b.size());
  brd_a.set(2, c.size());
  Eigen::IndexList<FixedOne, int, FixedOne> rsh_b;
  Eigen::IndexList<int, FixedOne, int> brd_b;
  brd_b.set(0, a.size());
  rsh_b.set(1, b.size());
  brd_b.set(2, c.size());
  Eigen::IndexList<FixedOne, FixedOne, int> rsh_c;
  Eigen::IndexList<int, int, FixedOne> brd_c;
  brd_c.set(0, a.size());
  brd_c.set(1, b.size());
  rsh_c.set(2, c.size());
  return a.reshape(rsh_a).broadcast(brd_a) + b.reshape(rsh_b).broadcast(brd_b) +
         c.reshape(rsh_c).broadcast(brd_c);
}

template <typename T>
inline decltype(auto) CollapseToVector(T &t)
{
  using Scalar = typename T::Scalar;
  Eigen::Map<Eigen::Matrix<Scalar, 1, Eigen::Dynamic>> mapped(
      t.data(),
      1,
      std::accumulate(
          t.dimensions().begin(), t.dimensions().end(), 1, std::multiplies<Eigen::Index>()));
  return mapped;
}

template <typename T, int toCollapse = 1>
inline decltype(auto) CollapseToMatrix(T &t)
{
  using Scalar = typename T::Scalar;
  Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> mapped(
      t.data(),
      std::accumulate(
          t.dimensions().begin(),
          t.dimensions().begin() + toCollapse,
          1,
          std::multiplies<Eigen::Index>()),
      std::accumulate(
          t.dimensions().begin() + toCollapse,
          t.dimensions().end(),
          1,
          std::multiplies<Eigen::Index>()));
  return mapped;
}

template <typename T, int toCollapse = 1>
inline decltype(auto) CollapseToMatrix(T const &t)
{
  using Scalar = typename T::Scalar;
  Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> const> mapped(
      t.data(),
      std::accumulate(
          t.dimensions().begin(),
          t.dimensions().begin() + toCollapse,
          1,
          std::multiplies<Eigen::Index>()),
      std::accumulate(
          t.dimensions().begin() + toCollapse,
          t.dimensions().end(),
          1,
          std::multiplies<Eigen::Index>()));
  return mapped;
}
