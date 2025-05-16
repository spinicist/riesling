#pragma once

#define LIBCUDACXX_ENABLE_SIMPLIFIED_COMPLEX_OPERATIONS
#include <cuda/std/complex>
#include <cuda/std/mdspan>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/universal_vector.h>

template <typename T> using CuCx = cuda::std::complex<T>;

// #define USE_BF16

#if defined USE_FP16
#warning "Enabled FP16"
using TDev = __half;
#define FLOAT_TO __float2half
#define FLOAT_FROM __hald2float
#define SHORT_TO __short2hald_rn
#define ONE CUDART_ONE_FP16
#define ZERO CUDART_ZERO_FP16
#elif defined USE_BF16
#warning "Enabled BF16"
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

inline constexpr struct ConvertToCx
{
  __host__ CuCx<TDev> operator()(std::complex<float> const z) { return CuCx<TDev>(FLOAT_TO(z.real()), FLOAT_TO(z.imag())); }
} ConvertToCx;

inline constexpr struct ConvertFromCx
{
  __host__ std::complex<float> operator()(CuCx<TDev> const z) const
  {
    return std::complex<float>(FLOAT_FROM(z.real()), FLOAT_FROM(z.imag()));
  }
} ConvertFromCx;

inline constexpr struct ConvertTo
{
  __host__ TDev operator()(float const f) const { return FLOAT_TO(f); }
} ConvertTo;

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

  HTensor(Span const s) // Construct a new tensor with the same shape as the input tensor
    : vec(s.size())
    , span(thrust::raw_pointer_cast(vec.data()), s.extents())
  {
  }

  auto size() -> size_t { return vec.size(); }
};

template <typename T, int N> struct UTensor
{
  using Vector = thrust::universal_vector<T>;
  using Extents = cuda::std::dextents<int, N>;
  using Span = cuda::std::mdspan<T, Extents, cuda::std::layout_left>;

  Vector vec;
  Span   span;

  template <typename... E> UTensor(E... e)
    : vec((e * ...))
    , span(thrust::raw_pointer_cast(vec.data()), e...)
  {
  }

  UTensor(Span const s) // Construct a new tensor with the same shape as the input tensor
    : vec(s.size())
    , span(thrust::raw_pointer_cast(vec.data()), s.extents())
  {
  }

  auto size() -> size_t { return vec.size(); }
};
