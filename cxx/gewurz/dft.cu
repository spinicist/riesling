#include "inputs.hpp"

#include "rl/io/hd5.hpp"
#include "rl/log.hpp"
#include "rl/types.hpp"

#include <cuda/std/complex>
#include <cuda/std/mdspan>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/sort.h>
#include <thrust/universal_vector.h>

using namespace rl;

template <int N> using ExtN = cuda::std::dextents<int, N>;
using Ext3 = ExtN<3>;
using Ext4 = ExtN<4>;

template <int N> using ReSN = cuda::std::mdspan<float, ExtN<N>, cuda::std::layout_left>;
using ReS3 = ReSN<3>;
using ReS4 = ReSN<4>;

template <int N> using CxSN = cuda::std::mdspan<cuda::std::complex<float>, ExtN<N>, cuda::std::layout_left>;
using CxS3 = CxSN<3>;
using CxS4 = CxSN<4>;

void ForwardDFT(CxS4 imgs, ReS3 traj, CxS3 ks)
{
  Log::Print("gewurz", "Forward DFT");
  int const nS = ks.extent(1);
  int const nT = ks.extent(2);
  int const nST = nS * nT;

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
          cuda::std::complex<float> const ep(cuda::std::cos(-phase), cuda::std::sin(-phase));
          for (int ic = 0; ic < ks.extent(0); ic++) {
            ks(ic, is, it) += ep * imgs(ii, ij, ik, ic);
          }
        }
      }
    }
  });
}

void AdjointDFT(CxS3 ks, ReS3 traj, CxS4 imgs)
{
  Log::Print("gewurz", "Adjoint DFT");
  int const nI = imgs.extent(0);
  int const nJ = imgs.extent(1);
  int const nK = imgs.extent(2);
  int const nIJ = nI * nJ;
  int const nIJK = nIJ * nK;

  thrust::for_each_n(thrust::host, thrust::make_counting_iterator(0), nIJK, [imgs, traj, ks] __host__ (int ijk) {
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
        cuda::std::complex<float> const ep(cuda::std::cos(p), cuda::std::sin(p));
        // fmt::print(stderr, "ijk {} {} {} st {} {} p {} ep {} {}\n", ii, ij, ik, is, it, p, ep.real(), ep.imag());
        for (int ic = 0; ic < ks.extent(0); ic++) {
          imgs(ii, ij, ik, ic) += ep * ks(ic, is, it);
        }
      }
    }
  });
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

  Log::Print("gewurz", "Create universal vectors");
  thrust::universal_vector<cuda::std::complex<float>> vKS(nC * nS * nT);
  auto                                                sKS = CxS3(thrust::raw_pointer_cast(vKS.data()), nC, nS, nT);
  thrust::universal_vector<cuda::std::complex<float>> vImg(mat[0] * mat[1] * mat[2] * nC);
  cuda::std::fill_n(vImg.data(), vImg.size(), 0.f);
  auto sImg = CxS4(thrust::raw_pointer_cast(vImg.data()), mat[0], mat[1], mat[2], nC);

  thrust::universal_vector<float> T(3 * nS * nT);
  thrust::copy_n(traj.points().data(), T.size(), T.data());
  auto sT = ReS3(thrust::raw_pointer_cast(T.data()), 3, nS, nT);
  Log::Print("gewurz", "Copy trajectory to device");
  AdjointDFT(sKS, sT, sImg);
  HD5::Writer writer(coreArgs.oname.Get());
  writer.writeInfo(info);
  writer.writeTensor(HD5::Keys::Data, AddBack(mat, nC, 1, 1), (Cx *)thrust::raw_pointer_cast(vImg.data()), HD5::Dims::Channels);
}
