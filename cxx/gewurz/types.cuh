#pragma once

#define LIBCUDACXX_ENABLE_SIMPLIFIED_COMPLEX_OPERATIONS
#include <cuda/std/complex>
#include <cuda/std/mdspan>
#include <cuda_bf16.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

template <typename T> using CuCx = cuda::std::complex<T>;
using THost = float;

#ifdef USE_BF16
using TDev = __nv_bfloat16;
#define FLOAT_TO __float2bfloat16
#define FLOAT_FROM __bfloat162float
#define SHORT_TO __short2bfloat16_rn
#define ONE CUDART_ONE_BF16
#define ZERO CUDART_ZERO_BF16
#else
using TDev = float;
#define FLOAT_TO
#define FLOAT_FROM
#define SHORT_TO
#define ONE 1.f
#define ZERO 0.f
#endif

struct ConvertToCx
{
  __host__ __device__ CuCx<TDev> operator()(CuCx<THost> const z) const { return CuCx<TDev>(FLOAT_TO(z.real()), FLOAT_TO(z.imag())); }
};

struct ConvertFromCx
{
  __host__ __device__ CuCx<THost> operator()(CuCx<TDev> const z) const { return CuCx<THost>(FLOAT_FROM(z.real()), FLOAT_FROM(z.imag())); }
};

struct ConvertTo
{
  __host__ __device__ TDev operator()(float const f) const
  {
    return FLOAT_TO(f);
  }
};


template <typename T, int N> struct DTensor
{
  using Vector = thrust::device_vector<T>;
  using Extents = cuda::std::dextents<int, N>;
  using Span = cuda::std::mdspan<T, Extents, cuda::std::layout_left>;

  Vector vec;
  Span   span;

  template <typename... E> DTensor(E... e)
    : vec((e * ...))
    , span(thrust::raw_pointer_cast(vec.data()), e...)
  {
  }

  DTensor(Span const s) // Construct a new tensor with the same shape as the input tensor
    : vec(s.size())
    , span(thrust::raw_pointer_cast(vec.data()), s.extents())
  {
  }

  auto size() -> size_t { return vec.size(); }
};

template <typename T, int N> struct HTensor
{
  using Vector = thrust::host_vector<T>;
  using Extents = cuda::std::dextents<int, N>;
  using Span = cuda::std::mdspan<T, Extents, cuda::std::layout_left>;

  Vector vec;
  Span   span;

  template <typename... E> HTensor(E... e)
    : vec((e * ...))
    , span(thrust::raw_pointer_cast(vec.data()), e...)
  {
  }

  auto size() -> size_t { return vec.size(); }
};
