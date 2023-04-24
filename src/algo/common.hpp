#pragma once

#include "log.hpp"
#include "tensorOps.hpp"

namespace rl {

template <typename Dims>
void CheckDimsEqual(Dims const a, Dims const b)
{
  if (a != b) {
    Log::Fail("Dimensions mismatch {} != {}", a, b);
  }
}

template <typename T>
inline float CheckedDot(T const &x1, T const &x2)
{
  Cx const dot = x1.dot(x2);
  constexpr float tol = 1.e-6f;
  if (std::abs(dot.imag()) > std::abs(dot.real()) * tol) {
    Log::Fail("Imaginary part of dot product {} exceeded {} times real part {}", dot.imag(), tol, dot.real());
  } else if (!std::isfinite(dot.real())) {
    Log::Fail("Dot product was not finite.");
  } else {
    return dot.real();
  }
}

}