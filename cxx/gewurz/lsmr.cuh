#pragma once

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
  MulPacked(MTensor mm)
    : m(mm) {};
  void forward(XTensor x, XTensor y) const override
  {
    int N = m.extent(0);
    for (int ii = 1; ii < m.rank(); ii++) {
      N *= m.extent(ii);
    }
    FTensor const  fx(x.data_handle(), x.extent(0), N);
    FTensor const  fy(y.data_handle(), y.extent(0), N);
    FMTensor const fm(this->m.data_handle(), N);
    auto           it = thrust::make_counting_iterator(0);
    thrust::for_each_n(thrust::cuda::par, it, N, [fx, fy, fm] __device__(int ii) {
      for (int ic = 0; ic < fx.extent(0); ic++) {
        fy(ic, ii) = fx(ic, ii) * fm(ii);
      }
    });
  }
  void adjoint(XTensor x, XTensor y) const override {}
};

template <typename T, int xRank, int yRank> struct LSMR
{
  struct Opts
  {
    int   imax = 4;
    float aTol = 1.e-6f;
    float bTol = 1.e-6f;
    float cTol = 1.e-6f;
    float λ = 0.f;
  };

  Op<T, xRank, yRank>::Ptr A;
  Op<T, yRank, yRank>::Ptr Minv = nullptr; // Left Pre-conditioner
  Op<T, xRank, xRank>::Ptr Ninv = nullptr; // Right Pre-conditioner

  Opts opts;

  auto run(DTensor<T, yRank>::Span const b, DTensor<T, xRank>::Span x) const;
};

} // namespace gw
