#include "bidiag.cuh"

namespace gw {
auto CuDot(thrust::device_vector<CuCxF> const &a, thrust::device_vector<CuCxF> const &b) -> float
{
  if (a.size() != b.size()) { throw rl::Log::Failure("Algo", "Dot product vectors had size {} and {}", a.size(), b.size()); }
  auto const  dot = thrust::inner_product(a.begin(), a.end(), b.begin(), CuCxF(0.f), thrust::plus<CuCxF>(),
                                          [] __device__(CuCxF const a, CuCxF const b) { return a * cuda::std::conj(b); });
  float const tol = 1.e-6;
  if (std::abs(dot.imag()) > std::abs(dot.real()) * tol) {
    throw rl::Log::Failure("Algo", "Imaginary part of dot product {} exceeded {} times real part {}", dot.imag(), tol,
                           dot.real());
  } else if (!std::isfinite(dot.real())) {
    throw rl::Log::Failure("Algo", "Dot product was not finite.");
  } else {
    return dot.real();
  }
}

auto CuNorm(thrust::device_vector<CuCxF> const &a) -> float { return std::sqrt(CuDot(a, a)); }
} // namespace gw
