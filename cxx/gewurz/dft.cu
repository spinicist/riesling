#include "inputs.hpp"

#include "rl/io/hd5.hpp"
#include "rl/log.hpp"
#include "rl/types.hpp"

#define LIBCUDACXX_ENABLE_SIMPLIFIED_COMPLEX_OPERATIONS
#include "math_constants.h"
#include <cuda/std/complex>
#include <cuda/std/mdspan>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/universal_vector.h>

using namespace rl;

template <int N> using ExtN = cuda::std::dextents<int, N>;
using Ext3 = ExtN<3>;
using Ext4 = ExtN<4>;

template <int N> using ReSN = cuda::std::mdspan<float, ExtN<N>, cuda::std::layout_left>;
using ReS3 = ReSN<3>;
using ReS4 = ReSN<4>;

using CCx = cuda::std::complex<float>;
template <int N> using CxSN = cuda::std::mdspan<CCx, ExtN<N>, cuda::std::layout_left>;
using CxS3 = CxSN<3>;
using CxS4 = CxSN<4>;

using ReVec = thrust::universal_vector<float>;
using CxVec = thrust::universal_vector<CCx>;

void ForwardDFT(CxS4 imgs, ReS3 traj, CxS3 ks)
{
  Log::Print("gewurz", "Forward DFT");
  auto const start = Log::Now();
  int const  nS = ks.extent(1);
  int const  nT = ks.extent(2);
  int const  nST = nS * nT;

  thrust::fill_n(ks.data_handle(), ks.size(), 0.f);
  thrust::for_each_n(thrust::cuda::par, thrust::make_counting_iterator(0), nST, [imgs, traj, ks] __device__(int st) {
    int const nI = imgs.extent(0);
    int const nJ = imgs.extent(1);
    int const nK = imgs.extent(2);
    int const nS = ks.extent(1);
    int const nT = ks.extent(2);

    int const it = st / nS;
    int const is = st % nS;

    for (int ii = 0; ii < nI; ii++) {
      for (int ij = 0; ij < nJ; ij++) {
        for (int ik = 0; ik < nK; ik++) {
          float const phase = 2.f * CUDART_PI_F * traj(0, is, it) * (ii - nI / 2.f) / nI +
                              2.f * CUDART_PI_F * traj(1, is, it) * (ij - nJ / 2.f) / nJ +
                              2.f * CUDART_PI_F * traj(2, is, it) * (ik - nK / 2.f) / nK;
          CCx const ep(cuda::std::cos(-phase), cuda::std::sin(-phase));
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

  Log::Print("gewurz", "Adjoint DFT {} {} {} {}", nI, nJ, nK, nIJK);
  auto const start = Log::Now();
  thrust::fill_n(imgs.data_handle(), imgs.size(), 0.f);
  thrust::for_each_n(thrust::cuda::par, thrust::make_counting_iterator(0), nIJK, [imgs, traj, ks] __device__(int ijk) {
    int const nI = imgs.extent(0);
    int const nJ = imgs.extent(1);
    int const nK = imgs.extent(2);
    int const nIJ = nI * nJ;
    int const nS = ks.extent(1);
    int const nT = ks.extent(2);

    int const ik = ijk / nIJ;
    int const ij = ijk % nIJ / nI;
    int const ii = ijk % nIJ % nI;

    for (int it = 0; it < nT; it++) {
      for (int is = 0; is < nS; is++) {
        float const p = 2.f * CUDART_PI_F * traj(0, is, it) * (ii - nI / 2.f) / nI +
                        2.f * CUDART_PI_F * traj(1, is, it) * (ij - nJ / 2.f) / nJ +
                        2.f * CUDART_PI_F * traj(2, is, it) * (ik - nK / 2.f) / nK;
        CCx const ep(cuda::std::cos(p), cuda::std::sin(p));
        // fmt::print(stderr, "ijk {} {} {} st {} {} p {} ep {} {}\n", ii, ij, ik, is, it, p, ep.real(), ep.imag());
        for (int ic = 0; ic < ks.extent(0); ic++) {
          imgs(ii, ij, ik, ic) += ep * ks(ic, is, it);
          // imgs(ii, ij, ik, ic) = ic + 1;
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

  Log::Print("gewurz", "Adjoint DFT {} {} {} {}", nI, nJ, nK, nIJK);
  auto const start = Log::Now();
  thrust::fill_n(imgs.data_handle(), imgs.size(), 0.f);
  thrust::for_each_n(thrust::cuda::par, thrust::make_counting_iterator(0), nIJK, [ks, M, traj, imgs] __device__(int ijk) {
    int const nI = imgs.extent(0);
    int const nJ = imgs.extent(1);
    int const nK = imgs.extent(2);
    int const nIJ = nI * nJ;
    int const nS = ks.extent(1);
    int const nT = ks.extent(2);

    int const ik = ijk / nIJ;
    int const ij = ijk % nIJ / nI;
    int const ii = ijk % nIJ % nI;

    for (int it = 0; it < nT; it++) {
      for (int is = 0; is < nS; is++) {
        float const p = 2.f * CUDART_PI_F * traj(0, is, it) * (ii - nI / 2.f) / nI +
                        2.f * CUDART_PI_F * traj(1, is, it) * (ij - nJ / 2.f) / nJ +
                        2.f * CUDART_PI_F * traj(2, is, it) * (ik - nK / 2.f) / nK;
        CCx const ep(cuda::std::cos(p), cuda::std::sin(p));
        // fmt::print(stderr, "ijk {} {} {} st {} {} p {} ep {} {}\n", ii, ij, ik, is, it, p, ep.real(), ep.imag());
        for (int ic = 0; ic < ks.extent(0); ic++) {
          imgs(ii, ij, ik, ic) += ep * ks(ic, is, it) * M(0, is, it);
          // imgs(ii, ij, ik, ic) = ic + 1;
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
  auto const  basis = LoadBasis(coreArgs.basisFile.Get());
  Trajectory  traj(reader, info.voxel_size, coreArgs.matrix.Get());
  auto const  mat = traj.matrix();
  Cx5 const   input = reader.readTensor<Cx5>();

  Log::Print("gewurz", "Allocate device memory");
  ReVec T(3 * nS * nT);
  thrust::copy_n(traj.points().data(), T.size(), T.data());
  auto sT = ReS3(thrust::raw_pointer_cast(T.data()), 3, nS, nT);

  CxVec vKS(nC * nS * nT);
  auto  sKS = CxS3(thrust::raw_pointer_cast(vKS.data()), nC, nS, nT);
  CxVec vImg(mat[0] * mat[1] * mat[2] * nC);
  auto  sImg = CxS4(thrust::raw_pointer_cast(vImg.data()), mat[0], mat[1], mat[2], nC);
  ReVec vM(1 * nS * nT);
  auto  sM = ReS3(thrust::raw_pointer_cast(vM.data()), 1, nS, nT);
  CxVec vMKS(1 * nS * nT);
  auto  sMKS = CxS3(thrust::raw_pointer_cast(vMKS.data()), 1, nS, nT);

  Log::Print("gewurz", "Poor man's SDC");
  thrust::fill(vMKS.begin(), vMKS.end(), 1.f);
  AdjointDFT(sMKS, sT, sImg);
  ForwardDFT(sImg, sT, sMKS);
  thrust::transform(thrust::cuda::par, vMKS.begin(), vMKS.end(), vM.begin(),
                    [] __device__(CCx x) { return 1.f / cuda::std::abs(x); });

  Log::Print("gewurz", "Recon");
  thrust::copy_n(input.data(), nC * nS * nT, vKS.begin());
  AdjointDFT(sKS, sM, sT, sImg);

  Cx6 output(mat[0], mat[1], mat[2], nC, 1, 1);
  thrust::copy_n(vImg.begin(), vImg.size(), output.data());
  HD5::Writer writer(coreArgs.oname.Get());
  writer.writeInfo(info);
  writer.writeTensor(HD5::Keys::Data, output.dimensions(), output.data(), HD5::Dims::Channels);
}
