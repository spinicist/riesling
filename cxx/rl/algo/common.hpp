#pragma once

#include "../log/log.hpp"
#include "../types.hpp"
#include "../sys/threads.hpp"

namespace rl {

template <typename Dims> void CheckDimsEqual(Dims const a, Dims const b)
{
  if (a != b) { throw Log::Failure("Algo", "Dimensions mismatch {} != {}", a, b); }
}

template <typename T> inline auto PairwiseDot(T const &x1, T const &x2, Index const st, Index const sz) -> typename T::Scalar
{
  if (sz < 128) {
    return x1.segment(st, sz).dot(x2.segment(st, sz));
  } else {
    auto const mid = sz / 2;
    return PairwiseDot(x1, x2, st, mid) + PairwiseDot(x1, x2, st + mid, sz - mid);
  }
}

template <typename Derived>
inline auto ParallelDot(Eigen::MatrixBase<Derived> const &x1, Eigen::MatrixBase<Derived> const &x2) ->
  typename Eigen::MatrixBase<Derived>::Scalar
{
  using Scalar = typename Eigen::MatrixBase<Derived>::Scalar;
  if (x1.size() != x2.size()) { throw Log::Failure("Algo", "Dot product vectors had size {} and {}", x1.size(), x2.size()); }
  auto const sz = x1.size();
  if (sz == 0) {
    return Scalar(0);
  } else {
    Index const                   nT = Threads::GlobalThreadCount();
    Index const                   den = sz / nT;
    Index const                   rem = sz % nT;
    Index const                   nC = std::min<Index>(sz, nT);
    Eigen::Barrier                barrier(nC);
    typename Derived::PlainObject partials(nC);
    for (Index ic = 0; ic < nC; ic++) {
      Index const lo = ic * den + std::min(ic, rem);
      Index const hi = (ic + 1) * den + std::min(ic + 1, rem);
      Index const n = hi - lo;
      Threads::GlobalPool()->Schedule([&x1, &x2, &barrier, &partials, ic, lo, n] {
        partials(ic) = PairwiseDot(x1, x2, lo, n);
        barrier.Notify();
      });
    }
    barrier.Wait();
    return partials.sum();
  }
}

template <typename T> inline auto CheckedDot(T const &x1, T const &x2) -> typename T::RealScalar
{
  if (x1.size() != x2.size()) { throw Log::Failure("Algo", "Dot product vectors had size {} and {}", x1.size(), x2.size()); }
  Cx const                     dot = ParallelDot(x1, x2);
  typename T::RealScalar const tol = 1.e-6;
  if (std::abs(dot.imag()) > std::abs(dot.real()) * tol) {
    throw Log::Failure("Algo", "Imaginary part of dot product {} exceeded {} times real part {}", dot.imag(), tol, dot.real());
  } else if (!std::isfinite(dot.real())) {
    throw Log::Failure("Algo", "Dot product was not finite.");
  } else {
    return dot.real();
  }
}

template <typename T> inline auto PairwiseNorm(T const &x, Index st = 0, Index sz = -1) -> typename T::RealScalar
{
  if (sz < 0) { sz = x.size(); }
  if (sz < 128) {
    return x.segment(st, sz).norm();
  } else {
    auto const mid = sz / 2;
    return PairwiseNorm(x, st, mid) + PairwiseNorm(x, st + mid, sz - mid);
  }
}

template <typename Derived>
inline auto ParallelNorm(Eigen::MatrixBase<Derived> const &v) -> typename Eigen::MatrixBase<Derived>::RealScalar
{
  Index const nT = Threads::GlobalThreadCount();
  if (v.rows() == 0) {
    return 0.f;
  } else {
    Index const nC = std::min<Index>(v.size(), nT);
    Index const den = v.size() / nC;
    Index const rem = v.size() % nC;

    typename Derived::PlainObject norms(nC);
    Eigen::Barrier                barrier(nC);

    for (Index ic = 0; ic < nC; ic++) {
      Index const lo = ic * den + std::min(ic, rem);
      Index const hi = (ic + 1) * den + std::min(ic + 1, rem);
      Index const n = hi - lo;
      Threads::GlobalPool()->Schedule([&v, &norms, &barrier, ic, lo, n] {
        norms[ic] = v.segment(lo, n).stableNorm();
        barrier.Notify();
      });
    }
    barrier.Wait();

    return norms.norm();
  }
}

} // namespace rl