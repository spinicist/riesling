#pragma once

#include "op.cuh"
#include "rl/log.hpp"
#include "types.cuh"
#include <thrust/inner_product.h>

namespace gw {

auto CuDot(thrust::device_vector<CuCxF> const &a, thrust::device_vector<CuCxF> const &b) -> float;
auto CuNorm(thrust::device_vector<CuCxF> const &a) -> float;

template <typename T, typename U> void CuScale(thrust::device_vector<T> &a, U const b)
{
  thrust::transform(thrust::cuda::par, a.begin(), a.end(), a.begin(), [b] __device__(T const a) { return a * b; });
}

template <typename T, typename U> void CuAsubBC2B(thrust::device_vector<T> &a, thrust::device_vector<T> &b, U const c)
{
  thrust::transform(a.begin(), a.end(), b.begin(), b.begin(),
                    [c] __device__(CuCxF const a, CuCxF const b) { return a - c * b; });
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
    , u(b)
    , Mu(b)
    , v(x)
    , Nv(x)
  {
    if (Minv) { Mu = DTensor<T, yRank>(u); }
    if (Ninv) { Nv = DTensor<T, xRank>(v); }

    thrust::fill(x.vec.begin(), x.vec.end(), CuCxF(0.f));
    if (Minv) {
      thrust::copy(b.vec.begin(), b.vec.end(), Mu.vec.begin());
      Minv->forward(Mu.span, u.span);
      β = std::sqrt(CuDot(Mu.vec, u.vec));
      CuScale(Mu.vec, 1.f / β);
    } else {
      thrust::copy(b.vec.begin(), b.vec.end(), u.vec.begin());
      β = std::sqrt(CuDot(Mu.vec, u.vec));
    }
    CuScale(u.vec, 1.f / β);

    if (Ninv) {
      A->adjoint(u.span, Nv.span);
      Ninv->forward(Nv.span, v.span);
      α = std::sqrt(CuDot(Nv.vec, v.vec));
      CuScale(Nv.vec, 1.f / α);
    } else {
      A->adjoint(u.span, v.span);
      α = std::sqrt(CuDot(v.vec, v.vec));
    }
    CuScale(v.vec, 1.f / α);
  }

  void next()
  {
    if (Minv) {
      A->forward(v.span, u.span);
      CuAsubBC2B(u.vec, Mu.vec, α);
      Minv->forward(Mu.span, u.span);
      β = std::sqrt(CuDot(Mu.vec, u.vec));
      CuScale(Mu.vec, 1.f / β);
    } else {
      A->forward(v.span, Mu.span);
      CuAsubBC2B(Mu.vec, u.vec, α);
      β = std::sqrt(CuDot(u.vec, u.vec));
    }
    CuScale(u.vec, 1.f / β);

    if (Ninv) {
      A->adjoint(u.span, v.span);
      CuAsubBC2B(v.vec, Nv.vec, β);
      Ninv->forward(Nv.span, v.span);
      α = std::sqrt(CuDot(Nv.vec, v.vec));
      CuScale(Nv.vec, 1.f / α);
    } else {
      A->adjoint(u.span, Nv.span);
      CuAsubBC2B(Nv.vec, v.vec, β);
      α = std::sqrt(CuDot(v.vec, v.vec));
    }
    CuScale(v.vec, 1.f / α);
  }
};
} // namespace gw
