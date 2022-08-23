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
  } else if (!std::isfinite(dot.real())) {
    Log::Fail("Dot product was not finite.");
  } else {
    return dot.real();
  }
}

inline auto SymOrtho(float const a, float const b)
{
  if (b == 0.f) {
    return std::make_tuple(std::copysign(1.f, a), 0.f, std::abs(a));
  } else if (a == 0.f) {
    return std::make_tuple(0.f, std::copysign(1.f, b), std::abs(b));
  } else if (std::abs(b) > std::abs(a)) {
    auto const τ = a / b;
    float s = std::copysign(1.f, b) / std::sqrt(1.f + τ * τ);
    return std::make_tuple(s, s * τ, b / s);
  } else {
    auto const τ = b / a;
    float c = std::copysign(1.f, a) / std::sqrt(1.f + τ * τ);
    return std::make_tuple(c, c * τ, a / c);
  }
}

} // namespace

} // namespace rl
