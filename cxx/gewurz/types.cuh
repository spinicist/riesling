#pragma once

#define LIBCUDACXX_ENABLE_SIMPLIFIED_COMPLEX_OPERATIONS
#include <cuda/std/complex>
#include <cuda/std/mdspan>
#include <cuda_fp16.h>
#include <math_constants.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/universal_vector.h>

using RealType = float;

template <int N> using ExtN = cuda::std::dextents<int, N>;
using Ext3 = ExtN<3>;
using Ext4 = ExtN<4>;

template <int N> using ReSN = cuda::std::mdspan<RealType, ExtN<N>, cuda::std::layout_left>;
using ReS3 = ReSN<3>;
using ReS4 = ReSN<4>;

using CCx = cuda::std::complex<RealType>;
template <int N> using CxSN = cuda::std::mdspan<CCx, ExtN<N>, cuda::std::layout_left>;
using CxS3 = CxSN<3>;
using CxS4 = CxSN<4>;

using ReVec = thrust::universal_vector<RealType>;
using CxVec = thrust::universal_vector<CCx>;

template <typename T, int N> struct DeviceTensor
{
  using Vector = thrust::device_vector<T>;
  using Extents = cuda::std::dextents<int, N>;
  using Span = cuda::std::mdspan<T, Extents, cuda::std::layout_left>;

  Vector vec;
  Span   span;

  template<typename... E>
  DeviceTensor(E... e)
    : vec((e * ...))
    , span(thrust::raw_pointer_cast(vec.data()), e...)
  {
  }

  auto size() -> size_t { return vec.size(); }
};

template <typename T, int N> struct HostTensor
{
  using Vector = thrust::host_vector<T>;
  using Extents = cuda::std::dextents<int, N>;
  using Span = cuda::std::mdspan<T, Extents, cuda::std::layout_left>;

  Vector vec;
  Span   span;


  template<typename... E>
  HostTensor(E... e)
    : vec((e * ...))
    , span(thrust::raw_pointer_cast(vec.data()), e...)
  {
  }

  auto size() -> size_t { return vec.size(); }
};
