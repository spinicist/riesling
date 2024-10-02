#pragma once

#include "log.hpp"
#include "tensors.hpp"

namespace rl {

template <typename Dims> void CheckDimsEqual(Dims const a, Dims const b)
{
  if (a != b) { throw Log::Failure("Algo", "Dimensions mismatch {} != {}", a, b); }
}

inline float ParNorm(Eigen::VectorXcf::MapType const &v)
{
  Index const        nT = Threads::GlobalThreadCount();
  if (v.rows() == 0) {
    return 0.f;
  } else {
    Index const    nC = std::min<Index>(v.size(), nT);
    Index const    den = v.size() / nC;
    Index const    rem = v.size() % nC;

    Eigen::VectorXf norms(nC);
    Eigen::Barrier barrier(nC);

    for (Index ic = 0; ic < nC; ic++) {
      Index const        lo = ic * den + std::min(ic, rem);
      Index const        hi = (ic + 1) * den + std::min(ic + 1, rem);
      Index const        n = hi - lo;
      Threads::GlobalPool()->Schedule([&] {
        norms[ic] = v.segment(lo, n).norm();
        barrier.Notify();
      });
    }
    barrier.Wait();

    return norms.norm();
  }
}

inline float ParNorm(Eigen::VectorXcf const &v)
{
  Index const        nT = Threads::GlobalThreadCount();
  if (v.rows() == 0) {
    return 0.f;
  } else {
    Index const    nC = std::min<Index>(v.size(), nT);
    Index const    den = v.size() / nC;
    Index const    rem = v.size() % nC;

    Eigen::VectorXf norms(nC);
    Eigen::Barrier barrier(nC);

    for (Index ic = 0; ic < nC; ic++) {
      Index const        lo = ic * den + std::min(ic, rem);
      Index const        hi = (ic + 1) * den + std::min(ic + 1, rem);
      Index const        n = hi - lo;
      Threads::GlobalPool()->Schedule([&v, &norms, &barrier, ic, lo, n] {
        norms[ic] = v.segment(lo, n).norm();
        barrier.Notify();
      });
    }
    barrier.Wait();

    return norms.norm();
  }
}

// Pairwise summation for accuracy
template <typename T> inline auto RecursiveDot(T const &x1, T const &x2, Index const st, Index const sz) -> typename T::Scalar
{
  if (sz < 8) {
    return x1.segment(st, sz).dot(x2.segment(st, sz));
  } else {
    auto const mid = sz / 2;
    return RecursiveDot(x1, x2, st, mid) + RecursiveDot(x1, x2, st + mid, sz - mid);
  }
}

template <typename T> inline auto CheckedDot(T const &x1, T const &x2) -> float
{
  // Pairwise summation for accuracy
  if (x1.size() != x2.size()) { throw Log::Failure("Algo", "Dot product vectors had size {} and {}", x1.size(), x2.size()); }
  Cx const    dot = RecursiveDot(x1, x2, 0, x1.size());
  float const tol = 1.e-6f;
  if (std::abs(dot.imag()) > std::abs(dot.real()) * tol) {
    throw Log::Failure("Algo", "Imaginary part of dot product {} exceeded {} times real part {}", dot.imag(), tol, dot.real());
  } else if (!std::isfinite(dot.real())) {
    throw Log::Failure("Algo", "Dot product was not finite. |x1| {} |x2| {}", x1.stableNorm(), x2.stableNorm());
  } else {
    return dot.real();
  }
}

} // namespace rl