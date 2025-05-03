#pragma once

#define LIBCUDACXX_ENABLE_SIMPLIFIED_COMPLEX_OPERATIONS
#include <cuda/std/complex>
#include <cuda/std/mdspan>
#include <cuda_fp16.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

using CuReal = float;
using CuCxF = cuda::std::complex<float>;

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
