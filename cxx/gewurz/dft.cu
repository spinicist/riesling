#include "dft.cuh"

#include "rl/log.hpp"

#define LIBCUDACXX_ENABLE_SIMPLIFIED_COMPLEX_OPERATIONS
#include <cub/device/device_for.cuh>
#include <cuda/experimental/stream.cuh>
#include <math_constants.h>

#include "types.cuh"

namespace cudax = cuda::experimental;

namespace gw::DFT {

void ThreeD::forward(DTensor<CuCxF, 3>::Span imgs, DTensor<CuCxF, 2>::Span ks) const
{
  rl::Log::Print("DFT", "Forward DFT");
  auto const  start = rl::Log::Now();
  int const   nS = ks.extent(0);
  int const   nT = ks.extent(1);
  int const   nST = nS * nT;
  int const   nI = imgs.extent(0);
  int const   nJ = imgs.extent(1);
  int const   nK = imgs.extent(2);
  int const   nIJK = nI * nJ * nK;
  float const scale = 1.f / std::sqrt(nIJK);

  thrust::for_each_n(thrust::cuda::par, thrust::make_counting_iterator(0), nST,
                     [imgs, traj = this->traj, ks, scale] __device__(int st) {
                       CuReal const pi2 = CuReal(2.f * CUDART_PI_F);

                       int const    nS = ks.extent(0);
                       int const    nT = ks.extent(1);
                       int const    it = st / nS;
                       int const    is = st % nS;
                       float3 const k = traj(is, it);
                       ks(is, it) = 0.f;

                       int const nI = imgs.extent(0);
                       int const nJ = imgs.extent(1);
                       int const nK = imgs.extent(2);
                       for (int ik = 0; ik < nK; ik++) {
                         float const z = (ik - nK / 2.f) / (float)nK;
                         for (int ij = 0; ij < nJ; ij++) {
                           float const y = (ij - nJ / 2.f) / (float)nJ;
                           for (int ii = 0; ii < nI; ii++) {
                             float const x = (ii - nI / 2.f) / (float)nI;
                             float const p = pi2 * (k.x * x + k.y * y + k.z * z);
                             CuCxF const ep(scale * cuda::std::cos(-p), scale * cuda::std::sin(-p));
                             ks(is, it) += ep * imgs(ii, ij, ik);
                           }
                         }
                       }
                     });
  rl::Log::Print("DFT", "Forward DFT finished in {}", rl::Log::ToNow(start));
}

void ThreeD::adjoint(DTensor<CuCxF, 2>::Span ks, DTensor<CuCxF, 3>::Span imgs) const
{
  int const   nI = imgs.extent(0);
  int const   nJ = imgs.extent(1);
  int const   nK = imgs.extent(2);
  int const   nIJ = nI * nJ;
  int const   nIJK = nIJ * nK;
  float const scale = 1.f / cuda::std::sqrt(nIJK);

  rl::Log::Print("DFT", "Adjoint DFT {} {} -> {} {} {} T {} {} {}", ks.extent(0), ks.extent(1), nI, nJ, nK, traj.extent(0),
                 traj.extent(1), traj.extent(2));
  auto const start = rl::Log::Now();

  auto it = thrust::make_counting_iterator(0);
  thrust::for_each_n(thrust::cuda::par, it, nIJK, [ks, traj = this->traj, imgs, scale] __device__(int ijk) {
    CuReal const pi2 = CuReal(2.f * CUDART_PI_F);
    int const    nI = imgs.extent(0);
    int const    nJ = imgs.extent(1);
    int const    nK = imgs.extent(2);
    int const    nIJ = nI * nJ;
    int const    nS = ks.extent(0);
    int const    nT = ks.extent(1);

    int const ik = ijk / nIJ;
    int const ij = ijk % nIJ / nI;
    int const ii = ijk % nIJ % nI;

    imgs(ii, ij, ik) = 0.f;

    float3 const r{(ii - nI / 2.f) / (float)nI, (ij - nJ / 2.f) / (float)nJ, (ik - nK / 2.f) / (float)nK};

    for (int it = 0; it < nT; it++) {
      for (int is = 0; is < nS; is++) {
        float3 const k = traj(is, it);
        float const  p = pi2 * (r.x * k.x + r.y * k.y + r.z * k.z);
        CuCxF const  ep(scale * cuda::std::cos(p), scale * cuda::std::sin(p));
        imgs(ii, ij, ik) += ep * ks(is, it);
      }
    }
  });
  rl::Log::Print("DFT", "Adjoint DFT finished in {}", rl::Log::ToNow(start));
}

template <int NP>
void ThreeDPacked<NP>::forward(DTensor<CuCxF, 4>::Span imgs, DTensor<CuCxF, 3>::Span ks) const
{
  if (NP != imgs.extent(0) || NP != ks.extent(0)) {
    throw rl::Log::Failure("DFT", "Packing dimension size mismatch");
  }
  rl::Log::Print("DFT", "Forward Packed DFT");
  auto const  start = rl::Log::Now();
  int const   nS = ks.extent(1);
  int const   nT = ks.extent(2);
  int const   nST = nS * nT;
  int const   nI = imgs.extent(1);
  int const   nJ = imgs.extent(2);
  int const   nK = imgs.extent(3);
  int const   nIJK = nI * nJ * nK;
  float const scale = 1.f / cuda::std::sqrt(nIJK);
  auto        it = thrust::make_counting_iterator(0);
  thrust::for_each_n(thrust::cuda::par, it, nST, [imgs, traj = this->traj, ks, scale] __device__(int st) {
    CuReal const pi2 = CuReal(2.f * CUDART_PI_F);
    int const    nC = ks.extent(0);
    int const    nS = ks.extent(1);
    int const    nT = ks.extent(2);
    int const    it = st / nS;
    int const    is = st % nS;
    float3 const k = traj(is, it);

    int const nI = imgs.extent(1);
    int const nJ = imgs.extent(2);
    int const nK = imgs.extent(3);
    CuCxF temp[NP] = {CuCxF(0.), };
    for (int ik = 0; ik < nK; ik++) {
      float const z = (ik - nK / 2.f) / (float)nK;
      for (int ij = 0; ij < nJ; ij++) {
        float const y = (ij - nJ / 2.f) / (float)nJ;
        for (int ii = 0; ii < nI; ii++) {
          float const x = (ii - nI / 2.f) / (float)nI;
          float const p = pi2 * (k.x * x + k.y * y + k.z * z);
          CuCxF const ep(scale * cuda::std::cos(-p), scale * cuda::std::sin(-p));
          for (int ic = 0; ic < NP; ic++) {
            temp[ic] += ep * imgs(ic, ii, ij, ik);
          }
        }
      }
    }
    for (int ic = 0; ic < NP; ic++) {
      ks(ic, is, it) = temp[ic];
    }
  });
  rl::Log::Print("DFT", "Forward Packed DFT finished in {}", rl::Log::ToNow(start));
}

template<int NP>
void ThreeDPacked<NP>::adjoint(DTensor<CuCxF, 3>::Span ks, DTensor<CuCxF, 4>::Span imgs) const
{
  if (NP != imgs.extent(0) || NP != ks.extent(0)) {
    throw rl::Log::Failure("DFT", "Packing dimension size mismatch");
  }
  int const   nI = imgs.extent(1);
  int const   nJ = imgs.extent(2);
  int const   nK = imgs.extent(3);
  int const   nIJ = nI * nJ;
  int const   nIJK = nIJ * nK;
  float const scale = 1.f / cuda::std::sqrt(nIJK);
  rl::Log::Print("DFT", "Adjoint Packed DFT {} {} {} -> {} {} {} {} T {} {} {}", ks.extent(0), ks.extent(1), ks.extent(2),
                 imgs.extent(0), nI, nJ, nK, traj.extent(0), traj.extent(1), traj.extent(2));
  auto const start = rl::Log::Now();

  auto it = thrust::make_counting_iterator(0);
  thrust::for_each_n(thrust::cuda::par, it, nIJK, [ks, traj = this->traj, imgs, scale] __device__(int ijk) {
    CuReal const pi2 = CuReal(2.f * CUDART_PI_F);
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

    CuCxF        temp[NP] = {CuCxF(0.f),};
    float3 const r{(ii - nI / 2.f) / (float)nI, (ij - nJ / 2.f) / (float)nJ, (ik - nK / 2.f) / (float)nK};
    for (int it = 0; it < nT; it++) {
      for (int is = 0; is < nS; is++) {
        float3 const k = traj(is, it);
        float const  p = pi2 * (r.x * k.x + r.y * k.y + r.z * k.z);
        CuCxF const  ep(scale * cuda::std::cos(p), scale * cuda::std::sin(p));
        for (int ic = 0; ic < NP; ic++) {
          temp[ic] += ep * ks(ic, is, it);
        }
      }
    }
    for (int ic = 0; ic < NP; ic++) {
      imgs(ic, ii, ij, ik) = temp[ic];
    }
  });
  rl::Log::Print("DFT", "Adjoint Packed DFT finished in {}", rl::Log::ToNow(start));
}

template struct ThreeDPacked<8>;

} // namespace gw::DFT
