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
  rl::Log::Print("gewurz", "Forward DFT");
  auto const start = rl::Log::Now();
  int const  nS = ks.extent(0);
  int const  nT = ks.extent(1);
  int const  nST = nS * nT;

  thrust::for_each_n(thrust::cuda::par, thrust::make_counting_iterator(0), nST,
                     [imgs, traj = this->traj, ks] __device__(int st) {
                       CuReal const pi2 = CuReal(2.f * CUDART_PI_F);

                       int const nI = imgs.extent(0);
                       int const nJ = imgs.extent(1);
                       int const nK = imgs.extent(2);
                       int const nS = ks.extent(0);
                       int const nT = ks.extent(1);

                       int const it = st / nS;
                       int const is = st % nS;

                       ks(is, it) = 0.f;
                       for (int ii = 0; ii < nI; ii++) {
                         for (int ij = 0; ij < nJ; ij++) {
                           for (int ik = 0; ik < nK; ik++) {
                             float const p = pi2 * (traj(0, is, it) * (CuReal)(ii - nI / 2) / (CuReal)nI +
                                                    traj(1, is, it) * (CuReal)(ij - nJ / 2) / (CuReal)nJ +
                                                    traj(2, is, it) * (CuReal)(ik - nK / 2) / (CuReal)nK);
                             CuCxF const ep(cuda::std::cos(-p), cuda::std::sin(-p));
                             ks(is, it) += ep * imgs(ii, ij, ik);
                           }
                         }
                       }
                     });
  rl::Log::Print("gewurz", "Forward DFT finished in {}", rl::Log::ToNow(start));
}

void ThreeD::adjoint(DTensor<CuCxF, 2>::Span ks, DTensor<CuCxF, 3>::Span imgs) const
{
  int const nI = imgs.extent(0);
  int const nJ = imgs.extent(1);
  int const nK = imgs.extent(2);
  int const nIJ = nI * nJ;
  int const nIJK = nIJ * nK;

  rl::Log::Print("gewurz", "Adjoint DFT {} {} -> {} {} {} T {} {} {}", ks.extent(0), ks.extent(1), nI, nJ, nK, traj.extent(0),
                 traj.extent(1), traj.extent(2));
  auto const start = rl::Log::Now();

  thrust::for_each_n(thrust::cuda::par, thrust::make_counting_iterator(0), nIJK,
                     [ks, traj = this->traj, imgs] __device__(int ijk) {
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

                       for (int it = 0; it < nT; it++) {
                         for (int is = 0; is < nS; is++) {
                           float const p = pi2 * (traj(0, is, it) * (CuReal)(ii - nI / 2) / (CuReal)nI +
                                                  traj(1, is, it) * (CuReal)(ij - nJ / 2) / (CuReal)nJ +
                                                  traj(2, is, it) * (CuReal)(ik - nK / 2) / (CuReal)nK);
                           CuCxF const ep(cuda::std::cos(p), cuda::std::sin(p));
                           imgs(ii, ij, ik) += ep * ks(is, it);
                         }
                       }
                     });
  rl::Log::Print("gewurz", "Adjoint DFT finished in {}", rl::Log::ToNow(start));
}

void ThreeDPacked::forward(DTensor<CuCxF, 4>::Span imgs, DTensor<CuCxF, 3>::Span ks) const
{
  rl::Log::Print("gewurz", "Forward Packed DFT");
  auto const start = rl::Log::Now();
  int const  nS = ks.extent(1);
  int const  nT = ks.extent(2);
  int const  nST = nS * nT;

  thrust::for_each_n(thrust::cuda::par, thrust::make_counting_iterator(0), nST,
                     [imgs, traj = this->traj, ks] __device__(int st) {
                       CuReal const pi2 = CuReal(2.f * CUDART_PI_F);
                       int const nC = imgs.extent(0);
                       int const nI = imgs.extent(1);
                       int const nJ = imgs.extent(2);
                       int const nK = imgs.extent(3);
                       int const nS = ks.extent(1);
                       int const nT = ks.extent(2);

                       int const it = st / nS;
                       int const is = st % nS;

                       for (int ic = 0; ic < nC; ic++) {
                         ks(ic, is, it) = 0.f;
                       }

                       for (int ii = 0; ii < nI; ii++) {
                         for (int ij = 0; ij < nJ; ij++) {
                           for (int ik = 0; ik < nK; ik++) {
                             float const p = pi2 * (traj(0, is, it) * (CuReal)(ii - nI / 2) / (CuReal)nI +
                                                    traj(1, is, it) * (CuReal)(ij - nJ / 2) / (CuReal)nJ +
                                                    traj(2, is, it) * (CuReal)(ik - nK / 2) / (CuReal)nK);
                             CuCxF const ep(cuda::std::cos(-p), cuda::std::sin(-p));
                             for (int ic = 0; ic < nC; ic++) {
                               ks(ic, is, it) += ep * imgs(ic, ii, ij, ik);
                             }
                           }
                         }
                       }
                     });
  rl::Log::Print("gewurz", "Forward Packed DFT finished in {}", rl::Log::ToNow(start));
}

void ThreeDPacked::adjoint(DTensor<CuCxF, 3>::Span ks, DTensor<CuCxF, 4>::Span imgs) const
{
  int const nI = imgs.extent(1);
  int const nJ = imgs.extent(2);
  int const nK = imgs.extent(3);
  int const nIJ = nI * nJ;
  int const nIJK = nIJ * nK;

  rl::Log::Print("gewurz", "Adjoint Packed DFT {} {} {} -> {} {} {} {}", ks.extent(0), ks.extent(1), ks.extent(2),
                 imgs.extent(0), nI, nJ, nK);
  auto const start = rl::Log::Now();
  auto       it = thrust::make_counting_iterator(0);
  thrust::for_each_n(thrust::cuda::par, it, nIJK, [ks, traj = this->traj, imgs] __device__(int ijk) {
    CuReal const pi2 = CuReal(2.f * CUDART_PI_F);
    int const nC = imgs.extent(0);
    int const nI = imgs.extent(1);
    int const nJ = imgs.extent(2);
    int const nK = imgs.extent(3);
    int const nIJ = nI * nJ;
    int const nS = ks.extent(1);
    int const nT = ks.extent(2);

    int const ik = ijk / nIJ;
    int const ij = ijk % nIJ / nI;
    int const ii = ijk % nIJ % nI;

    for (int ic = 0; ic < nC; ic++) {
      imgs(ic, ii, ij, ik) = 0.f;
    }

    for (int it = 0; it < nT; it++) {
      for (int is = 0; is < nS; is++) {
        float const p =
          pi2 * (traj(0, is, it) * (CuReal)(ii - nI / 2) / (CuReal)nI + traj(1, is, it) * (CuReal)(ij - nJ / 2) / (CuReal)nJ +
                 traj(2, is, it) * (CuReal)(ik - nK / 2) / (CuReal)nK);
        CuCxF const ep(cuda::std::cos(p), cuda::std::sin(p));
        for (int ic = 0; ic < nC; ic++) {
          imgs(ic, ii, ij, ik) += ep * ks(ic, is, it);
        }
      }
    }
  });
  rl::Log::Print("gewurz", "Adjoint Packed DFT finished in {}", rl::Log::ToNow(start));
}
} // namespace gw::DFT
