#define LIBCUDACXX_ENABLE_SIMPLIFIED_COMPLEX_OPERATIONS
#include "recon.cuh"

#include "rl/log/log.hpp"
#include <cub/device/device_for.cuh>
#include <cuda/experimental/stream.cuh>
#include <math_constants.h>

#include "types.cuh"

namespace cudax = cuda::experimental;

namespace gw {

  template<int NC>
  Recon<NC>::Recon(ST s, TT t)
    : sense(s)
    , traj(t) {

      if (NC != sense.extent(3)) { throw rl::Log::Failure("Recon", "SENSE maps had {} channels, expected {}", sense.extent(3), NC); }
    };


template <int NC> void Recon<NC>::forward(DTensor<CuCx<TDev>, 3>::Span img, DTensor<CuCx<TDev>, 3>::Span ks) const
{
  rl::Log::Print("RECON", "Forward starting");
  auto const start = rl::Log::Now();
  int const  nS = ks.extent(1);
  int const  nT = ks.extent(2);
  int const  nST = nS * nT;
  int const  nI = img.extent(0);
  int const  nJ = img.extent(1);
  int const  nK = img.extent(2);
  int const  nIJK = nI * nJ * nK;
  TDev const scale = FLOAT_TO(1.f / std::sqrt(nIJK));
  auto       it = thrust::make_counting_iterator(0);
  thrust::for_each_n(thrust::cuda::par, it, nST, [img, sense = this->sense, traj = this->traj, ks, scale] __device__(int st) {
    TDev const pi2 = FLOAT_TO(2.f * CUDART_PI_F);
    int const  nS = ks.extent(1);
    int const  nT = ks.extent(2);
    int const  it = st / nS;
    int const  is = st % nS;
    TDev const kx = traj(0, is, it);
    TDev const ky = traj(1, is, it);
    TDev const kz = traj(2, is, it);

    int const  nI = img.extent(0);
    int const  nJ = img.extent(1);
    int const  nK = img.extent(2);
    cuda::std::array<CuCx<TDev>, NC> temp{CuCx<TDev>{0.f, 0.f},};
    for (int ik = 0; ik < nK; ik++) {
      TDev const rz = FLOAT_TO((ik - nK / 2.f) / (float)nK);
      for (int ij = 0; ij < nJ; ij++) {
        TDev const ry = FLOAT_TO((ij - nJ / 2.f) / (float)nJ);
        for (int ii = 0; ii < nI; ii++) {
          TDev const       rx = FLOAT_TO((ii - nI / 2.f) / (float)nI);
          auto const       p = pi2 * (kx * rx + ky * ry + kz * rz);
          CuCx<TDev> const ep(cuda::std::cos(-p), cuda::std::sin(-p));
          for (int ic = 0; ic < NC; ic++) {
            temp[ic] += ep * sense(ii, ij, ik, ic) * img(ii, ij, ik);
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

template <int NC> void Recon<NC>::adjoint(DTensor<CuCx<TDev>, 3>::Span ks, DTensor<CuCx<TDev>, 3>::Span img) const
{
  int const  nI = img.extent(0);
  int const  nJ = img.extent(1);
  int const  nK = img.extent(2);
  int const  nIJ = nI * nJ;
  int const  nIJK = nIJ * nK;
  TDev const scale = FLOAT_TO(1.f / std::sqrt(nIJK));
  rl::Log::Print("RECON", "Adjoint starting");
  auto const start = rl::Log::Now();

  auto it = thrust::make_counting_iterator(0);
  thrust::for_each_n(thrust::cuda::par, it, nIJK, [ks, sense = this->sense, traj = this->traj, img, scale] __device__(int ijk) {
    TDev const pi2 = FLOAT_TO(2.f * CUDART_PI_F);
    int const  nI = img.extent(0);
    int const  nJ = img.extent(1);
    int const  nK = img.extent(2);
    int const  nIJ = nI * nJ;
    int const  nS = ks.extent(1);
    int const  nT = ks.extent(2);

    int const ik = ijk / nIJ;
    int const ij = ijk % nIJ / nI;
    int const ii = ijk % nIJ % nI;

    TDev const rx = FLOAT_TO((ii - nI / 2.f) / (float)nI);
    TDev const ry = FLOAT_TO((ij - nJ / 2.f) / (float)nJ);
    TDev const rz = FLOAT_TO((ik - nK / 2.f) / (float)nK);

    cuda::std::array<CuCx<TDev>, NC> temp{CuCx<TDev>{0.f, 0.f}};
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
    
    img(ii, ij, ik) = 0;
    for (int ic = 0; ic < NC; ic++) {
      img(ii, ij, ik) += scale * cuda::std::conj(sense(ii, ij, ik, ic)) * temp[ic];
    }
  });
  rl::Log::Print("DFT", "Adjoint Packed DFT finished in {}", rl::Log::ToNow(start));
}

template struct Recon<1>;
template struct Recon<8>;

} // namespace gw::DFT
