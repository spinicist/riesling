#pragma once

#include "../types.cuh"

#include <thrust/inner_product.h>

namespace gw {

template <typename T>
struct ConjMul
{};


template <>
struct ConjMul<CuCx<float>>
{
  __host__ __device__ CuCx<float> operator()(CuCx<float> const a, CuCx<float> const b) const { return a * cuda::std::conj(b); }
};

template <>
struct ConjMul<std::complex<float>>
{
  __host__ __device__ std::complex<float> operator()(std::complex<float> const a, std::complex<float> const b) const { return a * std::conj(b); }
};

template <typename T> auto CuDot(T const &a, T const &b) -> typename T::value_type::value_type;

auto CuNorm(thrust::device_vector<CuCx<TDev>> const &a) -> TDev;
}
