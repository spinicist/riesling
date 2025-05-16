#define LIBCUDACXX_ENABLE_SIMPLIFIED_COMPLEX_OPERATIONS
#include "dft.cuh"

#include "rl/log/log.hpp"
#include <cub/device/device_for.cuh>
#include <cuda/experimental/stream.cuh>
#include <math_constants.h>

#include "types.cuh"

namespace cudax = cuda::experimental;

namespace gw::DFT {

static constexpr int Sum = 16;

void ThreeD::forward(DTensor<CuCx<TDev>, 3>::Span imgs, DTensor<CuCx<TDev>, 2>::Span ks) const
{

  int const  nS = ks.extent(0);
  int const  nT = ks.extent(1);
  int const  nST = nS * nT;
  int const  nI = imgs.extent(0);
  int const  nJ = imgs.extent(1);
  int const  nK = imgs.extent(2);
  int const  nIJK = nI * nJ * nK;
  TDev const scale = FLOAT_TO(1.f / std::sqrt(nIJK));

  rl::Log::Print("DFT", "Forward DFT {} {} {} -> {} {} Scale {}", nI, nJ, nK, nS, nT, scale);
  auto const start = rl::Log::Now();

  auto it = thrust::make_counting_iterator<long>(0L);
  thrust::for_each_n(thrust::cuda::par, it, nST, [imgs, traj = this->traj, ks, scale] __device__(long st) {
    TDev const pi2 = FLOAT_TO(2.f * CUDART_PI_F);

    int const  nS = ks.extent(0);
    int const  nT = ks.extent(1);
    int const  it = st / nS;
    int const  is = st % nS;
    TDev const kx = traj(0, is, it);
    TDev const ky = traj(1, is, it);
    TDev const kz = traj(2, is, it);

    short const nI = imgs.extent(0);
    short const nJ = imgs.extent(1);
    short const nK = imgs.extent(2);

    CuCx<TDev> temp = ZERO;
    // long int   ind = 0;
    // ks(is, it) = 0;
    for (short ik = 0; ik < nK; ik++) {
      TDev const rz = FLOAT_TO((ik - nK / 2) / (float)nK);
      for (short ij = 0; ij < nJ; ij++) {
        TDev const ry = FLOAT_TO((ij - nJ / 2) / (float)nJ);
        for (short ii = 0; ii < nI; ii++) {
          TDev const       rx = FLOAT_TO((ii - nI / 2) / (float)nI);
          auto const       p = (kx * rx + ky * ry + kz * rz);
          CuCx<TDev> const ep(cuda::std::cos(-p), cuda::std::sin(-p));
          CuCx<TDev> const m = ep * imgs(ii, ij, ik);
          temp += m;

          if (is == 22 && it == 159 && ij == nJ/2 && ik == nK/2) {
            printf("st %ld is %d it %d i %d %d %d r %f %f %f p %f ep %f+%fi img %f+%fi m %f+%f temp %f+%fi\n", st, is, it, ii,
                   ij, ik, rx, ry, rz, p, ep.real(), ep.imag(), imgs(ii, ij, ik).real(), imgs(ii, ij, ik).imag(), m.real(),
                   m.imag(), temp.real(), temp.imag());
          }
          // ind++;
          // if (ind % Sum == 0) {
          //   ks(is, it) += scale * temp;
          //   temp = ZERO;
          // }
        }
      }
    }

    // if (blockIdx.x == 0 && threadIdx.x == 0 && ii == 0 && it == 0 && is == 0) {
    if (is == 22 && it == 159) {
      printf("st %ld is %d it %d scale %f temp %f %f sca %f\n", st, is, it, scale, temp.real(), temp.imag(),
             scale * cuda::std::abs(temp));
    }

    // if (ind % Sum != 0) { ks(is, it) += scale * temp; }
    ks(is, it) = scale * temp;
  });
  rl::Log::Print("DFT", "Forward DFT finished in {}", rl::Log::ToNow(start));
}

void ThreeD::adjoint(DTensor<CuCx<TDev>, 2>::Span ks, DTensor<CuCx<TDev>, 3>::Span imgs) const
{
  int const  nI = imgs.extent(0);
  int const  nJ = imgs.extent(1);
  int const  nK = imgs.extent(2);
  int const  nIJ = nI * nJ;
  int const  nIJK = nIJ * nK;
  TDev const scale = FLOAT_TO(1.f / cuda::std::sqrt(nIJK));
  rl::Log::Print("DFT", "Adjoint DFT {} {} -> {} {} {} Scale ", ks.extent(0), ks.extent(1), nI, nJ, nK, scale);
  auto const start = rl::Log::Now();
  auto       it = thrust::make_counting_iterator<long>(0L);
  thrust::for_each_n(thrust::cuda::par, it, nIJK, [ks, traj = this->traj, imgs, scale] __device__(long ijk) {
    TDev const pi2 = FLOAT_TO(2.f * CUDART_PI_F);
    int const  nI = imgs.extent(0);
    int const  nJ = imgs.extent(1);
    int const  nK = imgs.extent(2);
    int const  nIJ = nI * nJ;
    int const  nS = ks.extent(0);
    int const  nT = ks.extent(1);

    int const ik = ijk / nIJ;
    int const ij = ijk % nIJ / nI;
    int const ii = ijk % nIJ % nI;

    TDev const rx = FLOAT_TO((ii - nI / 2) / (float)nI);
    TDev const ry = FLOAT_TO((ij - nJ / 2) / (float)nJ);
    TDev const rz = FLOAT_TO((ik - nK / 2) / (float)nK);

    // printf("i %d %d %d r %4.3f %4.3f %4.3f\n", ii, ij, ik, rx, ry, rz);
    CuCx<TDev> temp = ZERO;
    long int   ind = 0;
    imgs(ii, ij, ik) = 0;
    for (int it = 0; it < nT; it++) {
      for (int is = 0; is < nS; is++) {
        TDev const       kx = traj(0, is, it);
        TDev const       ky = traj(1, is, it);
        TDev const       kz = traj(2, is, it);
        TDev const       p = (rx * kx + ry * ky + rz * kz);
        CuCx<TDev> const ep(cuda::std::cos(p), cuda::std::sin(p));
        temp += ep * ks(is, it);
        ind++;
        if (ind % Sum == 0) {
          imgs(ii, ij, ik) += scale * temp;
          temp = ZERO;
        }
      }
    }
    if (ind % Sum != 0) { imgs(ii, ij, ik) += scale * temp; }
    // imgs(ii, ij, ik) += scale * temp;
  });
  rl::Log::Print("DFT", "Adjoint DFT finished in {}", rl::Log::ToNow(start));
}

template <int NP> void ThreeDPacked<NP>::forward(DTensor<CuCx<TDev>, 4>::Span imgs, DTensor<CuCx<TDev>, 3>::Span ks) const
{
  if (NP != imgs.extent(0) || NP != ks.extent(0)) { throw rl::Log::Failure("DFT", "Packing dimension size mismatch"); }
  rl::Log::Print("DFT", "Forward Packed DFT");
  auto const start = rl::Log::Now();
  int const  nS = ks.extent(1);
  int const  nT = ks.extent(2);
  int const  nST = nS * nT;
  int const  nI = imgs.extent(1);
  int const  nJ = imgs.extent(2);
  int const  nK = imgs.extent(3);
  int const  nIJK = nI * nJ * nK;
  TDev const scale = FLOAT_TO(1.f / std::sqrt(nIJK));
  auto       it = thrust::make_counting_iterator(0);
  thrust::for_each_n(thrust::cuda::par, it, nST, [imgs, traj = this->traj, ks, scale] __device__(int st) {
    TDev const pi2 = FLOAT_TO(2.f * CUDART_PI_F);
    int const  nC = ks.extent(0);
    int const  nS = ks.extent(1);
    int const  nT = ks.extent(2);
    int const  it = st / nS;
    int const  is = st % nS;
    TDev const kx = traj(0, is, it);
    TDev const ky = traj(1, is, it);
    TDev const kz = traj(2, is, it);

    int const  nI = imgs.extent(1);
    int const  nJ = imgs.extent(2);
    int const  nK = imgs.extent(3);
    CuCx<TDev> temp[NP] = {
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

template <int NP> void ThreeDPacked<NP>::adjoint(DTensor<CuCx<TDev>, 3>::Span ks, DTensor<CuCx<TDev>, 4>::Span imgs) const
{
  if (NP != imgs.extent(0) || NP != ks.extent(0)) {
    throw rl::Log::Failure("DFT", "Packing dimension size mismatch. {} {} {}", NP, imgs.extent(0), ks.extent(0));
  }
  int const  nI = imgs.extent(1);
  int const  nJ = imgs.extent(2);
  int const  nK = imgs.extent(3);
  int const  nIJ = nI * nJ;
  int const  nIJK = nIJ * nK;
  TDev const scale = FLOAT_TO(1.f / std::sqrt(nIJK));
  rl::Log::Print("DFT", "Adjoint Packed DFT {} {} {} -> {} {} {} {} T {} {} {}", ks.extent(0), ks.extent(1), ks.extent(2),
                 imgs.extent(0), nI, nJ, nK, traj.extent(0), traj.extent(1), traj.extent(2));
  auto const start = rl::Log::Now();

  auto it = thrust::make_counting_iterator(0);
  thrust::for_each_n(thrust::cuda::par, it, nIJK, [ks, traj = this->traj, imgs, scale] __device__(int ijk) {
    TDev const pi2 = FLOAT_TO(2.f * CUDART_PI_F);
    int const  nC = imgs.extent(0);
    int const  nI = imgs.extent(1);
    int const  nJ = imgs.extent(2);
    int const  nK = imgs.extent(3);
    int const  nIJ = nI * nJ;
    int const  nS = ks.extent(1);
    int const  nT = ks.extent(2);

    int const ik = ijk / nIJ;
    int const ij = ijk % nIJ / nI;
    int const ii = ijk % nIJ % nI;

    CuCx<TDev> temp[NP] = {
      CuCx<TDev>(0.f),
    };

    TDev const rx = FLOAT_TO((ii - nI / 2.f) / (float)nI);
    TDev const ry = FLOAT_TO((ij - nJ / 2.f) / (float)nJ);
    TDev const rz = FLOAT_TO((ik - nK / 2.f) / (float)nK);

    // long int ind = 0;
    for (int it = 0; it < nT; it++) {
      for (int is = 0; is < nS; is++) {
        TDev const       kx = traj(0, is, it);
        TDev const       ky = traj(1, is, it);
        TDev const       kz = traj(2, is, it);
        TDev const       p = pi2 * ((rx * kx) + (ry * ky) + (rz * kz));
        CuCx<TDev> const ep(cuda::std::cos(p), cuda::std::sin(p));
        for (int ic = 0; ic < NP; ic++) {
          temp[ic] += ep * ks(ic, is, it);
        }
        // ind++;
        // if (ind % Sum == 0) {
        //   for (int ic = 0; ic < NP; ic++) {
        //     imgs(ic, ii, ij, ik) += scale * temp[ic];
        //     temp[ic] = CuCx<TDev>(0.f);
        //   }
        // }
      }
    }

    // if (ind % Sum != 0) {
    for (int ic = 0; ic < NP; ic++) {
      imgs(ic, ii, ij, ik) = scale * temp[ic];
    }
    // }
  });
  rl::Log::Print("DFT", "Adjoint Packed DFT finished in {}", rl::Log::ToNow(start));
}

template struct ThreeDPacked<1>;
template struct ThreeDPacked<8>;

} // namespace gw::DFT
