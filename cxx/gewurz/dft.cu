#include "inputs.hpp"

#include "rl/io/hd5.hpp"
#include "rl/log.hpp"
#include "rl/types.hpp"

#define LIBCUDACXX_ENABLE_SIMPLIFIED_COMPLEX_OPERATIONS
#include <cuda/std/complex>
#include <cuda/std/mdspan>
#include <cuda_fp16.h>
#include <math_constants.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/universal_vector.h>

#include "types.cuh"

using namespace rl;

void ForwardDFT(CxS4 imgs, ReS3 traj, CxS3 ks)
{
  Log::Print("gewurz", "Forward DFT");
  auto const start = Log::Now();
  int const  nS = ks.extent(1);
  int const  nT = ks.extent(2);
  int const  nST = nS * nT;

  thrust::for_each_n(thrust::cuda::par, thrust::make_counting_iterator(0), nST, [imgs, traj, ks] __device__(int st) {
    CuReal const pi2 = CuReal(2.f * CUDART_PI_F);

    int const nI = imgs.extent(0);
    int const nJ = imgs.extent(1);
    int const nK = imgs.extent(2);
    int const nS = ks.extent(1);
    int const nT = ks.extent(2);

    int const it = st / nS;
    int const is = st % nS;

    for (int ic = 0; ic < ks.extent(0); ic++) {
      ks(ic, is, it) = 0.f;
    }

    for (int ii = 0; ii < nI; ii++) {
      for (int ij = 0; ij < nJ; ij++) {
        for (int ik = 0; ik < nK; ik++) {
          float const p =
            pi2 * (traj(0, is, it) * (CuReal)(ii - nI / 2) / (CuReal)nI + traj(1, is, it) * (CuReal)(ij - nJ / 2) / (CuReal)nJ +
                   traj(2, is, it) * (CuReal)(ik - nK / 2) / (CuReal)nK);
          CuCx const ep(cuda::std::cos(-p), cuda::std::sin(-p));
          for (int ic = 0; ic < ks.extent(0); ic++) {
            ks(ic, is, it) += ep * imgs(ii, ij, ik, ic);
          }
        }
      }
    }
  });
  Log::Print("gewurz", "Forward DFT finished in {}", Log::ToNow(start));
}

void AdjointDFT(CxS3 ks, ReS3 traj, CxS4 imgs)
{
  int const nI = imgs.extent(0);
  int const nJ = imgs.extent(1);
  int const nK = imgs.extent(2);
  int const nIJ = nI * nJ;
  int const nIJK = nIJ * nK;

  Log::Print("gewurz", "Adjoint DFT {} {} {} -> {} {} {} {} T {} {} {}", ks.extent(0), ks.extent(1), ks.extent(2), nI, nJ, nK,
             imgs.extent(3), traj.extent(0), traj.extent(1), traj.extent(2));
  auto const start = Log::Now();
  thrust::for_each_n(thrust::cuda::par, thrust::make_counting_iterator(0), nIJK, [imgs, traj, ks] __device__(int ijk) {
    CuReal const pi2 = CuReal(2.f * CUDART_PI_F);

    int const nI = imgs.extent(0);
    int const nJ = imgs.extent(1);
    int const nK = imgs.extent(2);
    int const nIJ = nI * nJ;
    int const nS = ks.extent(1);
    int const nT = ks.extent(2);

    int const ik = ijk / nIJ;
    int const ij = ijk % nIJ / nI;
    int const ii = ijk % nIJ % nI;

    for (int ic = 0; ic < ks.extent(0); ic++) {
      imgs(ii, ij, ik, ic) = 0.f;
    }

    for (int it = 0; it < nT; it++) {
      for (int is = 0; is < nS; is++) {
        float const p =
          pi2 * (traj(0, is, it) * (CuReal)(ii - nI / 2) / (CuReal)nI + traj(1, is, it) * (CuReal)(ij - nJ / 2) / (CuReal)nJ +
                 traj(2, is, it) * (CuReal)(ik - nK / 2) / (CuReal)nK);
        CuCx const ep(cuda::std::cos(p), cuda::std::sin(p));
        for (int ic = 0; ic < ks.extent(0); ic++) {
          imgs(ii, ij, ik, ic) += ep * ks(ic, is, it);
        }
      }
    }
  });
  Log::Print("gewurz", "Adjoint DFT finished in {}", Log::ToNow(start));
}

void AdjointDFT(CxS3 ks, ReS3 M, ReS3 traj, CxS4 imgs)
{
  int const nI = imgs.extent(0);
  int const nJ = imgs.extent(1);
  int const nK = imgs.extent(2);
  int const nIJ = nI * nJ;
  int const nIJK = nIJ * nK;

  Log::Print("gewurz", "Adjoint DFT {} {} {} -> {} {} {} {} M {} {} {}", ks.extent(0), ks.extent(1), ks.extent(2), nI, nJ, nK,
             imgs.extent(3), M.extent(0), M.extent(1), M.extent(2));
  auto const start = Log::Now();
  thrust::for_each_n(thrust::cuda::par, thrust::make_counting_iterator(0), nIJK, [ks, M, traj, imgs] __device__(int ijk) {
    CuReal const pi2 = CuReal(2.f * CUDART_PI_F);

    int const nI = imgs.extent(0);
    int const nJ = imgs.extent(1);
    int const nK = imgs.extent(2);
    int const nIJ = nI * nJ;
    int const nS = ks.extent(1);
    int const nT = ks.extent(2);

    int const ik = ijk / nIJ;
    int const ij = ijk % nIJ / nI;
    int const ii = ijk % nIJ % nI;

    for (int ic = 0; ic < ks.extent(0); ic++) {
      imgs(ii, ij, ik, ic) = 0.f;
    }

    for (int it = 0; it < nT; it++) {
      for (int is = 0; is < nS; is++) {
        float const p =
          pi2 * (traj(0, is, it) * (CuReal)(ii - nI / 2) / (CuReal)nI + traj(1, is, it) * (CuReal)(ij - nJ / 2) / (CuReal)nJ +
                 traj(2, is, it) * (CuReal)(ik - nK / 2) / (CuReal)nK);
        CuCx const ep(cuda::std::cos(p), cuda::std::sin(p));
        for (int ic = 0; ic < ks.extent(0); ic++) {
          imgs(ii, ij, ik, ic) += ep * ks(ic, is, it) * M(0, is, it);
        }
      }
    }
  });
  Log::Print("gewurz", "Adjoint DFT finished in {}", Log::ToNow(start));
}

void main_dft(args::Subparser &parser)
{
  CoreArgs coreArgs(parser);
  ParseCommand(parser, coreArgs.iname, coreArgs.oname);

  HD5::Reader reader(coreArgs.iname.Get());
  Info const  info = reader.readInfo();
  auto const  shape = reader.dimensions();
  Index const nC = shape[0];
  Index const nS = shape[1];
  Index const nT = shape[2];

  Log::Print("gewurz", "Read trajectory");
  HostTensor<CuReal, 3> hT(3, nS, nT);
  reader.readTo(hT.vec.data(), HD5::Keys::Trajectory);
  auto const              mat = reader.readAttributeSz<3>(HD5::Keys::Trajectory, "matrix");
  DeviceTensor<CuReal, 3> T(3L, nS, nT);
  thrust::copy(hT.vec.begin(), hT.vec.end(), T.vec.data());

  Log::Print("gewurz", "Poor man's SDC");
  DeviceTensor<CuReal, 3> M(1L, nS, nT);
  DeviceTensor<CuCx, 3>   Mks(1L, nS, nT);
  DeviceTensor<CuCx, 4>   Mimgs(mat[0], mat[1], mat[2], 1);
  thrust::fill(Mks.vec.begin(), Mks.vec.end(), 1.f);
  AdjointDFT(Mks.span, T.span, Mimgs.span);
  ForwardDFT(Mimgs.span, T.span, Mks.span);
  thrust::transform(thrust::cuda::par, Mks.vec.begin(), Mks.vec.end(), M.vec.begin(),
                    [] __device__(CuCx x) { return (CuReal)1 / cuda::std::abs(x); });

  Log::Print("gewurz", "Recon");
  HostTensor<CuCx, 3>   hKS(nC, nS, nT);
  DeviceTensor<CuCx, 3> ks(nC, nS, nT);
  reader.readTo((Cx *)hKS.vec.data());
  thrust::copy(hKS.vec.begin(), hKS.vec.end(), ks.vec.begin());
  HostTensor<CuCx, 4>   hImgs(mat[0], mat[1], mat[2], nC);
  DeviceTensor<CuCx, 4> imgs(mat[0], mat[1], mat[2], nC);
  AdjointDFT(ks.span, M.span, T.span, imgs.span);
  thrust::copy(imgs.vec.begin(), imgs.vec.end(), hImgs.vec.begin());
  HD5::Writer writer(coreArgs.oname.Get());
  writer.writeInfo(info);
  writer.writeTensor(HD5::Keys::Data, Sz6{mat[0], mat[1], mat[2], nC, 1, 1}, (Cx *)hImgs.vec.data(), HD5::Dims::Channels);
}
