#pragma once

namespace rl {

namespace {

template <typename T>
inline float CheckedDot(T const &x1, T const &x2)
{
  Cx const dot = Dot(x1, x2);
  constexpr float tol = 1.e-6f;
  if (std::abs(dot.imag()) > std::abs(dot.real()) * tol) {
    Log::Fail("Imaginary part of dot product {} exceeded {} times real part {}", dot.imag(), tol, dot.real());
  } else {
    return dot.real();
  }
}

} // namespace

} // namespace rl
