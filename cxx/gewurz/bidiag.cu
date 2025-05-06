#include "bidiag.cuh"

namespace gw {

namespace {
struct ConjMul
{
  __host__ __device__ CuCxH operator()(CuCxH const a, CuCxH const b) const { return a * cuda::std::conj(b); }
};
} // namespace

auto CuDot(thrust::device_vector<CuCxH> const &a, thrust::device_vector<CuCxH> const &b) -> __half
{
  if (a.size() != b.size()) { throw rl::Log::Failure("Algo", "Dot product vectors had size {} and {}", a.size(), b.size()); }
  auto const dot =
    thrust::inner_product(a.begin(), a.end(), b.begin(), CuCxH(__float2half(0)), thrust::plus<CuCxH>(), ConjMul());
  __half const tol = __float2half(1.e-6f);
  if (cuda::std::abs(dot.imag()) > cuda::std::abs(dot.real()) * tol) {
    throw rl::Log::Failure("Algo", "Imaginary part of dot product {} exceeded {} times real part {}", __half2float(dot.imag()),
                           __half2float(tol), __half2float(dot.real()));
  } else if (!cuda::std::isfinite(dot.real())) {
    throw rl::Log::Failure("Algo", "Dot product was not finite");
  } else {
    return dot.real();
  }
}

auto CuNorm(thrust::device_vector<CuCxH> const &a) -> __half { return cuda::std::sqrt(CuDot(a, a)); }
} // namespace gw
