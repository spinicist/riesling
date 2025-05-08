#define CUB_DEBUG_SYNC
#define LIBCUDACXX_ENABLE_SIMPLIFIED_COMPLEX_OPERATIONS

#include "dft2.cuh"
#include "rl/log.hpp"
#include <cub/device/device_for.cuh>
#include <math_constants.h>

#include "types.cuh"

namespace cudax = cuda::experimental;

namespace gw::DFT {

void ThreeD2::forward(DTensor<CuCx<TDev>, 3>::Span imgs, DTensor<CuCx<TDev>, 2>::Span ks) const
{
  rl::Log::Print("DFT2", "Forward DFT");
  auto const start = rl::Log::Now();
  int const  nI = imgs.extent(0);
  int const  nJ = imgs.extent(1);
  int const  nK = imgs.extent(2);
  int const  nS = ks.extent(0);
  int const  nT = ks.extent(1);
  TDev const scale = FLOAT_TO(1.f / std::sqrt(nI * nJ * nK));
  thrust::fill_n(thrust::cuda::par, ks.data_handle(), ks.size(), CuCx<TDev>(0));
  // ForEachInExtents is rightmost
  cuda::std::dextents<int, 5> TSKJI(nT, nS, nK, nJ, nI);
  cub::DeviceFor::ForEachInExtents(
    TSKJI,
    [ks, traj = this->traj, imgs, scale] __device__(int ind, int it, int is, int ik, int ij, int ii) {
      TDev const pi2 = FLOAT_TO(2.f * CUDART_PI_F);

      int const nI = imgs.extent(0);
      int const nJ = imgs.extent(1);
      int const nK = imgs.extent(2);

      TDev const rx = FLOAT_TO((ii - nI / 2.f) / (float)nI);
      TDev const ry = FLOAT_TO((ij - nJ / 2.f) / (float)nJ);
      TDev const rz = FLOAT_TO((ik - nK / 2.f) / (float)nK);

      TDev const kx = traj(0, is, it);
      TDev const ky = traj(1, is, it);
      TDev const kz = traj(2, is, it);

      TDev const       p = pi2 * (rx * kx + ry * ky + rz * kz);
      CuCx<TDev> const ep(cuda::std::cos(-p), cuda::std::sin(-p));
      ks(is, it) += ep * imgs(ii, ij, ik);
    },
    stream.get());
  stream.wait();
  rl::Log::Print("DFT2", "Forward DFT finished in {}", rl::Log::ToNow(start));
}

void ThreeD2::adjoint(DTensor<CuCx<TDev>, 2>::Span ks, DTensor<CuCx<TDev>, 3>::Span imgs) const
{
  int const  nS = ks.extent(0);
  int const  nT = ks.extent(1);
  int const  nI = imgs.extent(0);
  int const  nJ = imgs.extent(1);
  int const  nK = imgs.extent(2);
  TDev const scale = FLOAT_TO(1.f / cuda::std::sqrt(nI * nJ * nK));

  rl::Log::Print("DFT2", "Adjoint DFT {} {} -> {} {} {} T {} {} {}", ks.extent(0), ks.extent(1), nI, nJ, nK, traj.extent(0),
                 traj.extent(1), traj.extent(2));
  auto const start = rl::Log::Now();
  thrust::fill_n(thrust::cuda::par, imgs.data_handle(), imgs.size(), CuCx<TDev>(0));
  // ForEachInExtents is rightmost
  cuda::std::dextents<long int, 5> KJITS(nK, nJ, nI, nT, nS);
  fmt::print(stderr, "KJITS {} {} {} {} {}\n", KJITS.extent(0), KJITS.extent(1), KJITS.extent(2), KJITS.extent(3),
             KJITS.extent(4));
  cub::DeviceFor::ForEachInExtents(
    KJITS,
    [ks, traj = this->traj, imgs, scale] __device__(long int ind, int ik, int ij, int ii, int it, int is) {
      TDev const pi2 = FLOAT_TO(2.f * CUDART_PI_F);
      int const  nI = imgs.extent(0);
      int const  nJ = imgs.extent(1);
      int const  nK = imgs.extent(2);

      TDev const rx = FLOAT_TO((ii - nI / 2.f) / (float)nI);
      TDev const ry = FLOAT_TO((ij - nJ / 2.f) / (float)nJ);
      TDev const rz = FLOAT_TO((ik - nK / 2.f) / (float)nK);

      TDev const kx = traj(0, is, it);
      TDev const ky = traj(1, is, it);
      TDev const kz = traj(2, is, it);

      TDev const       p = pi2 * (rx * kx + ry * ky + rz * kz);
      CuCx<TDev> const ep(cuda::std::cos(p), cuda::std::sin(p));
      imgs(ii, ij, ik) += scale * ep * ks(is, it);

      if (blockIdx.x == 0 && threadIdx.x == 0 && ii == 0 && it == 0 && is == 0) {
        printf("idx %ld is %d it %d ii %d ij %d ik %d ti %d %d %d bi %d %d %d gd %d %d %d bd %d %d %d\n", ind, is, it, ii, ij, ik, threadIdx.x,
               threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z, gridDim.x, gridDim.y, gridDim.z, blockDim.x,
               blockDim.y, blockDim.z);
      }
    },
    stream.get());
  stream.wait();
  rl::Log::Print("DFT2", "Adjoint DFT finished in {}", rl::Log::ToNow(start));
}

template <int NP> void ThreeDPacked2<NP>::forward(DTensor<CuCx<TDev>, 4>::Span imgs, DTensor<CuCx<TDev>, 3>::Span ks) const
{
  if (NP != imgs.extent(0) || NP != ks.extent(0)) { throw rl::Log::Failure("DFT", "Packing dimension size mismatch"); }
  rl::Log::Print("DFT2", "Forward Packed DFT");
  auto const start = rl::Log::Now();
  int const  nI = imgs.extent(1);
  int const  nJ = imgs.extent(2);
  int const  nK = imgs.extent(3);
  int const  nS = ks.extent(1);
  int const  nT = ks.extent(2);
  TDev const scale = FLOAT_TO(1.f / std::sqrt(nI * nJ * nK));
  thrust::fill_n(thrust::cuda::par, ks.data_handle(), ks.size(), CuCx<TDev>(0));
  // ForEachInExtents is rightmost
  cuda::std::dextents<int, 5> TSKJI(nT, nS, nK, nJ, nI);
  cub::DeviceFor::ForEachInExtents(
    TSKJI,
    [ks, traj = this->traj, imgs, scale] __device__(int ind, int it, int is, int ik, int ij, int ii) {
      TDev const pi2 = FLOAT_TO(2.f * CUDART_PI_F);

      int const nI = imgs.extent(0);
      int const nJ = imgs.extent(1);
      int const nK = imgs.extent(2);

      TDev const rx = FLOAT_TO((ii - nI / 2.f) / (float)nI);
      TDev const ry = FLOAT_TO((ij - nJ / 2.f) / (float)nJ);
      TDev const rz = FLOAT_TO((ik - nK / 2.f) / (float)nK);

      TDev const kx = traj(0, is, it);
      TDev const ky = traj(1, is, it);
      TDev const kz = traj(2, is, it);

      TDev const       p = pi2 * (rx * kx + ry * ky + rz * kz);
      CuCx<TDev> const ep(cuda::std::cos(-p), cuda::std::sin(-p));
      for (int ip = 0; ip < NP; ip++) {
        ks(ip, is, it) += ep * imgs(ip, ii, ij, ik);
      }
    },
    stream.get());
  stream.wait();
  rl::Log::Print("DFT2", "Forward Packed DFT finished in {}", rl::Log::ToNow(start));
}

template <int NP> void ThreeDPacked2<NP>::adjoint(DTensor<CuCx<TDev>, 3>::Span ks, DTensor<CuCx<TDev>, 4>::Span imgs) const
{
  if (NP != imgs.extent(0) || NP != ks.extent(0)) { throw rl::Log::Failure("DFT", "Packing dimension size mismatch"); }
  rl::Log::Print("DFT2", "Adjoint Packed DFT");
  int const  nS = ks.extent(1);
  int const  nT = ks.extent(2);
  int const  nI = imgs.extent(1);
  int const  nJ = imgs.extent(2);
  int const  nK = imgs.extent(3);
  TDev const scale = FLOAT_TO(1.f / cuda::std::sqrt(nI * nJ * nK));
  auto const start = rl::Log::Now();
  thrust::fill_n(thrust::cuda::par, imgs.data_handle(), imgs.size(), CuCx<TDev>(0));
  // ForEachInExtents is rightmost
  cuda::std::dextents<int, 5> KJITS(nK, nJ, nI, nT, nS);
  cub::DeviceFor::ForEachInExtents(
    KJITS,
    [ks, traj = this->traj, imgs, scale] __device__(int ind, int ik, int ij, int ii, int it, int is) {
      TDev const pi2 = FLOAT_TO(2.f * CUDART_PI_F);
      int const  nI = imgs.extent(0);
      int const  nJ = imgs.extent(1);
      int const  nK = imgs.extent(2);

      TDev const rx = FLOAT_TO((ii - nI / 2.f) / (float)nI);
      TDev const ry = FLOAT_TO((ij - nJ / 2.f) / (float)nJ);
      TDev const rz = FLOAT_TO((ik - nK / 2.f) / (float)nK);

      TDev const kx = traj(0, is, it);
      TDev const ky = traj(1, is, it);
      TDev const kz = traj(2, is, it);

      TDev const       p = pi2 * (rx * kx + ry * ky + rz * kz);
      CuCx<TDev> const ep(cuda::std::cos(p), cuda::std::sin(p));
      for (int ip = 0; ip < NP; ip++) {
        imgs(ip, ii, ij, ik) += scale * ep * ks(ip, is, it);
      }

      // if (blockIdx.x == 1 && threadIdx.x == 1) {
      //   printf("idx %d ii %d ij %d ik %d ti %d %d %d bi %d %d %d gd %d %d %d bd %d %d %d\n", ind, ii, ij, ik, threadIdx.x,
      //          threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z, gridDim.x, gridDim.y, gridDim.z, blockDim.x,
      //          blockDim.y, blockDim.z);
      // }
    },
    stream.get());
  stream.wait();
  rl::Log::Print("DFT", "Adjoint Packed DFT finished in {}", rl::Log::ToNow(start));
}

template struct ThreeDPacked2<8>;

} // namespace gw::DFT
