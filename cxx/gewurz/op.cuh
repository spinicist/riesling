#pragma once

#include "rl/log.hpp"
#include "types.cuh"

namespace gw {

template <typename T, int xRank, int yRank> struct Op
{
  virtual void forward(DTensor<T, xRank>::Span x, DTensor<T, yRank>::Span y) const;
  virtual void adjoint(DTensor<T, yRank>::Span y, DTensor<T, xRank>::Span x) const;
};

template <typename T, typename MT, int Rank> struct MulPacked : Op<T, Rank, Rank>
{
  using XTensor = DTensor<T, Rank>::Span;
  using MTensor = DTensor<MT, Rank - 1>::Span;
  using FTensor = DTensor<T, 2>::Span;
  using FMTensor = DTensor<MT, 1>::Span;
  MTensor const m;

  struct MulKernel {
    int M, N;
    __host__ __device__ __half operator()(float const f) const
    {
      return __float2half(f);
    }
  }

  MulPacked(MTensor mm)
    : m(mm) {};
  void forward(XTensor x, XTensor y) const override
  {
    int const M = x.extent(0);
    if (M != y.extent(0)) { throw rl::Log::Failure("Mul", "M {} y[0] {}", M, y.extent(0)); }
    int N = 1;
    for (int ii = 0; ii < m.rank(); ii++) {
      if (m.extent(ii) != x.extent(ii + 1)) {
        throw rl::Log::Failure("Mul", "Dim {} extent mismatch m {} x {}", ii, m.extent(ii), x.extent(ii + 1));
      }
      if (m.extent(ii) != y.extent(ii + 1)) {
        throw rl::Log::Failure("Mul", "Dim {} extent mismatch m {} y {}", ii, m.extent(ii), y.extent(ii + 1));
      }
      N *= m.extent(ii);
    }
    FTensor const  fx(x.data_handle(), M, N);
    FTensor const  fy(y.data_handle(), M, N);
    FMTensor const fm(this->m.data_handle(), N);
    rl::Log::Print("Mul", "Forward M {} N {}", M, N);
    auto const start = rl::Log::Now();
    auto       it = thrust::make_counting_iterator(0);
    thrust::for_each_n(thrust::cuda::par, it, N, [fx, fy, fm, M] __device__(int ii) {
      for (int ic = 0; ic < M; ic++) {
        fy(ic, ii) = fx(ic, ii) * fm(ii);
      }
    });
    rl::Log::Print("Mul", "Finished in {}", rl::Log::ToNow(start));
  }
  void adjoint(XTensor x, XTensor y) const override { throw rl::Log::Failure("Mul", "Adjoint not implemented"); }
};

} // namespace gw
