#include "dft.cuh"

#include "rl/log.hpp"

#define LIBCUDACXX_ENABLE_SIMPLIFIED_COMPLEX_OPERATIONS
#include <cub/device/device_for.cuh>
#include <cuda/experimental/stream.cuh>
#include <math_constants.h>

#include "types.cuh"

namespace cudax = cuda::experimental;

namespace gw::DFT {

void ThreeD::forward(DTensor<CuCxH, 3>::Span imgs, DTensor<CuCxH, 2>::Span ks) const
{
  rl::Log::Print("DFT", "Forward DFT");
  auto const   start = rl::Log::Now();
  int const    nS = ks.extent(0);
  int const    nT = ks.extent(1);
  int const    nST = nS * nT;
  int const    nI = imgs.extent(0);
  int const    nJ = imgs.extent(1);
  int const    nK = imgs.extent(2);
  int const    nIJK = nI * nJ * nK;
  __nv_bfloat16 const scale = __float2bfloat16(1.f / std::sqrt(nIJK));

  auto it = thrust::make_counting_iterator(0);
  thrust::for_each_n(thrust::cuda::par, it, nST,
                     [imgs, traj = this->traj, ks, scale] __device__(int st) {
                       __nv_bfloat16 const pi2 = __float2bfloat16(2.f * CUDART_PI_F);

                       int const    nS = ks.extent(0);
                       int const    nT = ks.extent(1);
                       int const    it = st / nS;
                       int const    is = st % nS;
                       __nv_bfloat16 const kx = traj(0, is, it);
                       __nv_bfloat16 const ky = traj(1, is, it);
                       __nv_bfloat16 const kz = traj(2, is, it);

                       short const nI = imgs.extent(0);
                       short const nJ = imgs.extent(1);
                       short const nK = imgs.extent(2);

                       __nv_bfloat16 const incx = CUDART_ONE_BF16 / __short2bfloat16_rn(nI);
                       __nv_bfloat16 const incy = CUDART_ONE_BF16 / __short2bfloat16_rn(nJ);
                       __nv_bfloat16 const incz = CUDART_ONE_BF16 / __short2bfloat16_rn(nK);

                       __nv_bfloat16 rx = -__float2bfloat16(0.5f);
                       __nv_bfloat16 ry = -__float2bfloat16(0.5f);
                       __nv_bfloat16 rz = -__float2bfloat16(0.5f);

                       CuCxH temp = CUDART_ZERO_BF16;
                       for (int ik = 0; ik < nK; ik++) {
                         for (int ij = 0; ij < nJ; ij++) {
                           for (int ii = 0; ii < nI; ii++) {
                             auto const   p = pi2 * (kx * rx + ky * ry + kz * rz);
                             CuCxH const  ep(cuda::std::cos(-p), cuda::std::sin(-p));
                             ks(is, it) += ep * imgs(ii, ij, ik);
                             rx += incx;
                           }
                           ry += incy;
                         }
                         rz += incz;
                       }
                       ks(is, it) = scale * temp;
                     });
  rl::Log::Print("DFT", "Forward DFT finished in {}", rl::Log::ToNow(start));
}

void ThreeD::adjoint(DTensor<CuCxH, 2>::Span ks, DTensor<CuCxH, 3>::Span imgs) const
{
  int const    nI = imgs.extent(0);
  int const    nJ = imgs.extent(1);
  int const    nK = imgs.extent(2);
  int const    nIJ = nI * nJ;
  int const    nIJK = nIJ * nK;
  __nv_bfloat16 const scale = __float2bfloat16(1.f / cuda::std::sqrt(nIJK));

  rl::Log::Print("DFT", "Adjoint DFT {} {} -> {} {} {} T {} {} {}", ks.extent(0), ks.extent(1), nI, nJ, nK, traj.extent(0),
                 traj.extent(1), traj.extent(2));
  auto const start = rl::Log::Now();

  auto it = thrust::make_counting_iterator(0);
  thrust::for_each_n(thrust::cuda::par, it, nIJK, [ks, traj = this->traj, imgs, scale] __device__(int ijk) {
    __nv_bfloat16 const pi2 = __float2bfloat16(2.f * CUDART_PI_F);
    int const    nI = imgs.extent(0);
    int const    nJ = imgs.extent(1);
    int const    nK = imgs.extent(2);
    int const    nIJ = nI * nJ;
    int const    nS = ks.extent(0);
    int const    nT = ks.extent(1);

    int const ik = ijk / nIJ;
    int const ij = ijk % nIJ / nI;
    int const ii = ijk % nIJ % nI;

    __nv_bfloat16 const rx = __float2bfloat16((ii - nI / 2.f) / (float)nI);
    __nv_bfloat16 const ry = __float2bfloat16((ij - nJ / 2.f) / (float)nJ);
    __nv_bfloat16 const rz = __float2bfloat16((ik - nK / 2.f) / (float)nK);

    CuCxH temp = CUDART_ZERO_BF16;
    for (int it = 0; it < nT; it++) {
      for (int is = 0; is < nS; is++) {
        __nv_bfloat16 const kx = traj(0, is, it);
        __nv_bfloat16 const ky = traj(1, is, it);
        __nv_bfloat16 const kz = traj(2, is, it);
        __nv_bfloat16 const   p = pi2 * (rx * kx + ry * ky + rz * kz);
        CuCxH const  ep(cuda::std::cos(p), cuda::std::sin(p));
        temp += ep * ks(is, it);
      }
    }
    imgs(ii, ij, ik) = scale * temp;
  });
  rl::Log::Print("DFT", "Adjoint DFT finished in {}", rl::Log::ToNow(start));
}

template <int NP> void ThreeDPacked<NP>::forward(DTensor<CuCxH, 4>::Span imgs, DTensor<CuCxH, 3>::Span ks) const
{
  if (NP != imgs.extent(0) || NP != ks.extent(0)) { throw rl::Log::Failure("DFT", "Packing dimension size mismatch"); }
  rl::Log::Print("DFT", "Forward Packed DFT");
  auto const   start = rl::Log::Now();
  int const    nS = ks.extent(1);
  int const    nT = ks.extent(2);
  int const    nST = nS * nT;
  int const    nI = imgs.extent(1);
  int const    nJ = imgs.extent(2);
  int const    nK = imgs.extent(3);
  int const    nIJK = nI * nJ * nK;
  __nv_bfloat16 const scale = __float2bfloat16(1.f / std::sqrt(nIJK));
  auto         it = thrust::make_counting_iterator(0);
  thrust::for_each_n(thrust::cuda::par, it, nST, [imgs, traj = this->traj, ks, scale] __device__(int st) {
    __nv_bfloat16 const pi2 = __float2bfloat16(2.f * CUDART_PI_F);
    int const    nC = ks.extent(0);
    int const    nS = ks.extent(1);
    int const    nT = ks.extent(2);
    int const    it = st / nS;
    int const    is = st % nS;
    __nv_bfloat16 const kx = traj(0, is, it);
    __nv_bfloat16 const ky = traj(1, is, it);
    __nv_bfloat16 const kz = traj(2, is, it);

    int const nI = imgs.extent(1);
    int const nJ = imgs.extent(2);
    int const nK = imgs.extent(3);
    CuCxH     temp[NP] = {
      CuCxH(0.),
    };
    for (int ik = 0; ik < nK; ik++) {
      __nv_bfloat16 const rz = __float2bfloat16((ik - nK / 2.f) / (float)nK);
      for (int ij = 0; ij < nJ; ij++) {
        __nv_bfloat16 const ry = __float2bfloat16((ij - nJ / 2.f) / (float)nJ);
        for (int ii = 0; ii < nI; ii++) {
          __nv_bfloat16 const rx = __float2bfloat16((ii - nI / 2.f) / (float)nI);
          auto const   p = pi2 * (kx * rx + ky * ry + kz * rz);
          CuCxH const  ep(cuda::std::cos(-p), cuda::std::sin(-p));
          for (int ic = 0; ic < NP; ic++) {
            temp[ic] += ep * imgs(ic, ii, ij, ik);
          }
        }
      }
    }
    for (int ic = 0; ic < NP; ic++) {
      ks(ic, is, it) = scale * temp[ic];
    }
  });
  rl::Log::Print("DFT", "Forward Packed DFT finished in {}", rl::Log::ToNow(start));
}

template <int NP> void ThreeDPacked<NP>::adjoint(DTensor<CuCxH, 3>::Span ks, DTensor<CuCxH, 4>::Span imgs) const
{
  if (NP != imgs.extent(0) || NP != ks.extent(0)) { throw rl::Log::Failure("DFT", "Packing dimension size mismatch"); }
  int const    nI = imgs.extent(1);
  int const    nJ = imgs.extent(2);
  int const    nK = imgs.extent(3);
  int const    nIJ = nI * nJ;
  int const    nIJK = nIJ * nK;
  __nv_bfloat16 const scale = __float2bfloat16(1.f / std::sqrt(nIJK));
  rl::Log::Print("DFT", "Adjoint Packed DFT {} {} {} -> {} {} {} {} T {} {} {}", ks.extent(0), ks.extent(1), ks.extent(2),
                 imgs.extent(0), nI, nJ, nK, traj.extent(0), traj.extent(1), traj.extent(2));
  auto const start = rl::Log::Now();

  auto it = thrust::make_counting_iterator(0);
  thrust::for_each_n(thrust::cuda::par, it, nIJK, [ks, traj = this->traj, imgs, scale] __device__(int ijk) {
    __nv_bfloat16 const pi2 = __float2bfloat16(2.f * CUDART_PI_F);
    int const    nC = imgs.extent(0);
    int const    nI = imgs.extent(1);
    int const    nJ = imgs.extent(2);
    int const    nK = imgs.extent(3);
    int const    nIJ = nI * nJ;
    int const    nS = ks.extent(1);
    int const    nT = ks.extent(2);

    int const ik = ijk / nIJ;
    int const ij = ijk % nIJ / nI;
    int const ii = ijk % nIJ % nI;

    CuCxH temp[NP] = {
      CuCxH(0.f),
    };

    __nv_bfloat16 const rx = __float2bfloat16((ii - nI / 2.f) / (float)nI);
    __nv_bfloat16 const ry = __float2bfloat16((ij - nJ / 2.f) / (float)nJ);
    __nv_bfloat16 const rz = __float2bfloat16((ik - nK / 2.f) / (float)nK);

    for (int it = 0; it < nT; it++) {
      for (int is = 0; is < nS; is++) {
        __nv_bfloat16 const kx = traj(0, is, it);
        __nv_bfloat16 const ky = traj(1, is, it);
        __nv_bfloat16 const kz = traj(2, is, it);
        auto const   p = pi2 * (rx * kx + ry * ky + rz * kz);
        CuCxH const  ep(cuda::std::cos(p), cuda::std::sin(p));
        for (int ic = 0; ic < NP; ic++) {
          temp[ic] += ep * ks(ic, is, it);
        }
      }
    }
    for (int ic = 0; ic < NP; ic++) {
      imgs(ic, ii, ij, ik) = scale * temp[ic];
    }
  });
  rl::Log::Print("DFT", "Adjoint Packed DFT finished in {}", rl::Log::ToNow(start));
}

template struct ThreeDPacked<8>;

} // namespace gw::DFT
