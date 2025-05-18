#pragma once

#include "../op/op.cuh"
#include "rl/log/log.hpp"
#include "types.cuh"
#include <thrust/inner_product.h>

namespace gw {

auto CuDot(thrust::device_vector<CuCx<TDev>> const &a, thrust::device_vector<CuCx<TDev>> const &b) -> TDev;
auto CuNorm(thrust::device_vector<CuCx<TDev>> const &a) -> TDev;

template <typename T, typename U> void CuScale(thrust::device_vector<T> &a, U const b)
{
  thrust::transform(thrust::cuda::par, a.begin(), a.end(), a.begin(), [b] __device__(T const a) { return a * b; });
}

template <typename T, typename U> void CuAsubBC(thrust::device_vector<T> const &a, thrust::device_vector<T> const &b, U const c, thrust::device_vector<T> &y)
{
  thrust::transform(a.begin(), a.end(), b.begin(), y.begin(), [c] __device__(T const a, T const b) { return a - b * c; });
}

struct RotationT
{
  float c, s, ρ;
};

template <typename T, int xRank, int yRank> struct Bidiag
{
  Op<T, xRank, yRank> const *A;
  Op<T, yRank, yRank> const *Minv = nullptr; // Left Pre-conditioner
  Op<T, xRank, xRank> const *Ninv = nullptr; // Right Pre-conditioner

  DTensor<T, yRank> u, Mu;
  DTensor<T, xRank> v, Nv;

  float α;
  float β;

  Bidiag(Op<T, xRank, yRank> const *Ain,
         Op<T, yRank, yRank> const *Mi,
         Op<T, xRank, xRank> const *Ni,
         DTensor<T, xRank>         &x,
         DTensor<T, yRank> const   &b)
    : A{Ain}
    , Minv{Mi}
    , Ninv{Ni}
    , u(b.span)
    , Mu(b.span)
    , v(x.span)
    , Nv(x.span)
  {
    thrust::fill(x.vec.begin(), x.vec.end(), T(0));
    if (Minv) {
      thrust::copy(b.vec.begin(), b.vec.end(), Mu.vec.begin());
      Minv->forward(Mu.span, u.span);
      β = FLOAT_FROM(cuda::std::sqrt(CuDot(Mu.vec, u.vec)));
      CuScale(Mu.vec, FLOAT_TO(1.f / β));
    } else {
      thrust::copy(b.vec.begin(), b.vec.end(), u.vec.begin());
      β = FLOAT_FROM(cuda::std::sqrt(CuDot(u.vec, u.vec)));
    }
    CuScale(u.vec, FLOAT_TO(1.f / β));

    if (Ninv) {
      A->adjoint(u.span, Nv.span);
      Ninv->forward(Nv.span, v.span);
      α = FLOAT_FROM(cuda::std::sqrt(CuDot(Nv.vec, v.vec)));
      CuScale(Nv.vec, FLOAT_TO(1 / α));
    } else {
      A->adjoint(u.span, v.span);
      α = FLOAT_FROM(cuda::std::sqrt(CuDot(v.vec, v.vec)));
    }
    CuScale(v.vec, FLOAT_TO(1.f / α));
  }

  void next()
  {
    if (Minv) {
      A->forward(v.span, u.span);
      CuAsubBC(u.vec, Mu.vec, FLOAT_TO(α), Mu.vec);
      Minv->forward(Mu.span, u.span);
      β = FLOAT_FROM(cuda::std::sqrt(CuDot(Mu.vec, u.vec)));
      CuScale(Mu.vec, FLOAT_TO(1.f / β));
    } else {
      A->forward(v.span, Mu.span);
      CuAsubBC(Mu.vec, u.vec, FLOAT_TO(α), u.vec);
      β = FLOAT_FROM(cuda::std::sqrt(CuDot(u.vec, u.vec)));
    }
    CuScale(u.vec, FLOAT_TO(1.f / β));

    if (Ninv) {
      A->adjoint(u.span, v.span);
      CuAsubBC(v.vec, Nv.vec, FLOAT_TO(β), Nv.vec);
      Ninv->forward(Nv.span, v.span);
      α = FLOAT_FROM(cuda::std::sqrt(CuDot(Nv.vec, v.vec)));
      CuScale(Nv.vec, FLOAT_TO(1.f / α));
    } else {
      A->adjoint(u.span, Nv.span);
      CuAsubBC(Nv.vec, v.vec, FLOAT_TO(β), v.vec);
      α = FLOAT_FROM(cuda::std::sqrt(CuDot(v.vec, v.vec)));
    }
    CuScale(v.vec, FLOAT_TO(1.f / α));
  }
};
} // namespace gw
