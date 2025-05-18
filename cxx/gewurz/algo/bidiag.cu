#include "bidiag.cuh"

namespace gw {

namespace {
struct ConjMul
{
  __host__ __device__ CuCx<TDev> operator()(CuCx<TDev> const a, CuCx<TDev> const b) const { return a * cuda::std::conj(b); }
};
} // namespace

auto CuDot(thrust::device_vector<CuCx<TDev>> const &a, thrust::device_vector<CuCx<TDev>> const &b) -> TDev
{
  if (a.size() != b.size()) { throw rl::Log::Failure("Algo", "Dot product vectors had size {} and {}", a.size(), b.size()); }
  auto const dot =
    thrust::inner_product(a.begin(), a.end(), b.begin(), CuCx<TDev>(FLOAT_TO(0)), thrust::plus<CuCx<TDev>>(), ConjMul());
  TDev const tol = FLOAT_TO(1.e-6f);
  if (cuda::std::abs(dot.imag()) > cuda::std::abs(dot.real()) * tol) {
    throw rl::Log::Failure("Algo", "Imaginary part of dot product {} exceeded {} times real part {}", FLOAT_FROM(dot.imag()),
                           FLOAT_FROM(tol), FLOAT_FROM(dot.real()));
  } else if (!cuda::std::isfinite(dot.real())) {
    throw rl::Log::Failure("Algo", "Dot product was not finite");
  } else {
    return dot.real();
  }
}

auto CuNorm(thrust::device_vector<CuCx<TDev>> const &a) -> TDev { return cuda::std::sqrt(CuDot(a, a)); }
} // namespace gw
