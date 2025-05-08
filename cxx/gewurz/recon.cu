#define LIBCUDACXX_ENABLE_SIMPLIFIED_COMPLEX_OPERATIONS
#include "dft.cuh"

#include "rl/log.hpp"
#include <cub/device/device_for.cuh>
#include <cuda/experimental/stream.cuh>
#include <math_constants.h>

#include "types.cuh"

namespace cudax = cuda::experimental;

namespace gw::DFT {

template <int NC> void Recon<NC>::forward(DTensor<CuCx<TDev>, 3>::Span img, DTensor<CuCx<TDev>, 3>::Span ks) const
{
  if (NC != imgs.extent(0) || NC != ks.extent(0)) { throw rl::Log::Failure("RECON", "Channel size mismatch"); }
  rl::Log::Print("RECON", "Forward starting");
  auto const start = rl::Log::Now();
  int const  nS = ks.extent(1);
  int const  nT = ks.extent(2);
  int const  nST = nS * nT;
  int const  nI = imgs.extent(0);
  int const  nJ = imgs.extent(1);
  int const  nK = imgs.extent(2);
  int const  nIJK = nI * nJ * nK;
  TDev const scale = FLOAT_TO(1.f / std::sqrt(nIJK));
  auto       it = thrust::make_counting_iterator(0);
  thrust::for_each_n(thrust::cuda::par, it, nST, [imgs, traj = this->traj, ks, scale] __device__(int st) {
    TDev const pi2 = FLOAT_TO(2.f * CUDART_PI_F);
    int const  nS = ks.extent(1);
    int const  nT = ks.extent(2);
    int const  it = st / nS;
    int const  is = st % nS;
    TDev const kx = traj(0, is, it);
    TDev const ky = traj(1, is, it);
    TDev const kz = traj(2, is, it);

    int const  nI = imgs.extent(0);
    int const  nJ = imgs.extent(1);
    int const  nK = imgs.extent(2);
    CuCx<TDev> temp[NC] = {
      CuCx<TDev>(0.),
    };
    for (int ik = 0; ik < nK; ik++) {
      TDev const rz = FLOAT_TO((ik - nK / 2.f) / (float)nK);
      for (int ij = 0; ij < nJ; ij++) {
        TDev const ry = FLOAT_TO((ij - nJ / 2.f) / (float)nJ);
        for (int ii = 0; ii < nI; ii++) {
          TDev const       rx = FLOAT_TO((ii - nI / 2.f) / (float)nI);
          auto const       p = pi2 * (kx * rx + ky * ry + kz * rz);
          CuCx<TDev> const ep(cuda::std::cos(-p), cuda::std::sin(-p));
          for (int ic = 0; ic < NC; ic++) {
            temp[ic] += ep * sense(ic, ii, ij, ik) * imgs(ii, ij, ik);
          }
        }
      }
    }
    for (int ic = 0; ic < NC; ic++) {
      ks(ic, is, it) = scale * temp[ic];
    }
  });
  rl::Log::Print("RECON", "Forward finished in {}", rl::Log::ToNow(start));
}

template <int NC> void Recon<NC>::adjoint(DTensor<CuCx<TDev>, 3>::Span ks, DTensor<CuCx<TDev>, 3>::Span imgs) const
{
  if (NC != imgs.extent(0) || NC != ks.extent(0)) { throw rl::Log::Failure("DFT", "Packing dimension size mismatch"); }
  int const  nI = imgs.extent(0);
  int const  nJ = imgs.extent(1);
  int const  nK = imgs.extent(2);
  int const  nIJ = nI * nJ;
  int const  nIJK = nIJ * nK;
  TDev const scale = FLOAT_TO(1.f / std::sqrt(nIJK));
  rl::Log::Print("RECON", "Adjoint starting");
  auto const start = rl::Log::Now();

  auto it = thrust::make_counting_iterator(0);
  thrust::for_each_n(thrust::cuda::par, it, nIJK, [ks, traj = this->traj, imgs, scale] __device__(int ijk) {
    TDev const pi2 = FLOAT_TO(2.f * CUDART_PI_F);
    int const  nI = imgs.extent(0);
    int const  nJ = imgs.extent(1);
    int const  nK = imgs.extent(2);
    int const  nIJ = nI * nJ;
    int const  nS = ks.extent(1);
    int const  nT = ks.extent(2);

    int const ik = ijk / nIJ;
    int const ij = ijk % nIJ / nI;
    int const ii = ijk % nIJ % nI;

    TDev const rx = FLOAT_TO((ii - nI / 2.f) / (float)nI);
    TDev const ry = FLOAT_TO((ij - nJ / 2.f) / (float)nJ);
    TDev const rz = FLOAT_TO((ik - nK / 2.f) / (float)nK);

    std::array<CuCx<TDev>, NC> temp{CuCx<TDev>(0.f)};
    for (int it = 0; it < nT; it++) {
      for (int is = 0; is < nS; is++) {
        TDev const       kx = traj(0, is, it);
        TDev const       ky = traj(1, is, it);
        TDev const       kz = traj(2, is, it);
        TDev const       p = pi2 * ((rx * kx) + (ry * ky) + (rz * kz));
        CuCx<TDev> const ep(cuda::std::cos(p), cuda::std::sin(p));
        for (int ic = 0; ic < NC; ic++) {
          temp[ic] += ep * ks(ic, is, it);
        }
      }
    }
    CuCx<TDev> const s = cuda::std::conj(sense(ic, ii, ij, ik));
    for (int ic = 0; ic < NC; ic++) {
      imgs(ic, ii, ij, ik) = scale * s temp[ic];
    }
  });
  rl::Log::Print("DFT", "Adjoint Packed DFT finished in {}", rl::Log::ToNow(start));
}

template struct ThreeDPacked<8>;

} // namespace gw::DFT
