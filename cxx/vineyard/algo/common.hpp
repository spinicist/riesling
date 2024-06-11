#pragma once

#include "log.hpp"
#include "tensors.hpp"

namespace rl {

template <typename Dims>
void CheckDimsEqual(Dims const a, Dims const b)
{
  if (a != b) { Log::Fail("Dimensions mismatch {} != {}", a, b); }
}

// Pairwise summation for accuracy
template <typename T>
inline auto RecursiveDot(T const &x1, T const &x2, Index const st, Index const sz) -> typename T::Scalar
{
  if (sz < 8) {
    return x1.segment(st, sz).dot(x2.segment(st, sz));
  } else {
    auto const mid = sz / 2;
    return RecursiveDot(x1, x2, st, mid) + RecursiveDot(x1, x2, st + mid, sz - mid);
  }
}

template <typename T>
inline auto CheckedDot(T const &x1, T const &x2) -> float
{
  // Pairwise summation for accuracy
  if (x1.size() != x2.size()) { Log::Fail("Dot product vectors had size {} and {}", x1.size(), x2.size()); }
  Cx const    dot = RecursiveDot(x1, x2, 0, x1.size());
  float const tol = 1.e-6f;
  if (std::abs(dot.imag()) > std::abs(dot.real()) * tol) {
    Log::Fail("Imaginary part of dot product {} exceeded {} times real part {}", dot.imag(), tol, dot.real());
  } else if (!std::isfinite(dot.real())) {
    Log::Fail("Dot product was not finite. |x1| {} |x2| {}", x1.stableNorm(), x2.stableNorm());
  } else {
    return dot.real();
  }
}

} // namespace rl