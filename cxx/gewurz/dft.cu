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

using Ext3 = cuda::std::dextents<int, 3>;
using Ext4 = cuda::std::dextents<int, 4>;

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
  auto sKS = cuda::std::mdspan(thrust::raw_pointer_cast(vKS.data()), Ext3(nC, nS, nT));
  thrust::universal_vector<cuda::std::complex<float>> vImg(mat[0] * mat[1] * mat[2] * nC);
  auto sImg = cuda::std::mdspan(thrust::raw_pointer_cast(vImg.data()), Ext4(mat[0], mat[1], mat[2], nC));

  thrust::universal_vector<float> vtKS(3 * nS * nT);
  thrust::universal_vector<float> vtImg(3 * mat[0] * mat[1] * mat[2]);
  auto                            stKS = cuda::std::mdspan(thrust::raw_pointer_cast(vtKS.data()), Ext3(3, nS, nT));
  Log::Print("gewurz", "Copy trajectory to device");
  thrust::copy_n(traj.points().data(), stKS.size(), stKS.data_handle());
  Log::Print("gewurz", "Set up image space Fourier indices");
  auto stImg = cuda::std::mdspan<float, Ext4>(thrust::raw_pointer_cast(vtImg.data()), Ext4(3, mat[0], mat[1], mat[2]));
  for (Index ik = 0; ik < mat[2]; ik++) {
    float const fk = (2.f * M_PI * (ik - mat[2] / 2)) / mat[2];
    for (Index ij = 0; ij < mat[1]; ij++) {
      float const fj = (2.f * M_PI * (ij - mat[1] / 2)) / mat[1];
      for (Index ii = 0; ii < mat[0]; ii++) {
        float const fi = (2.f * M_PI * (ii - mat[0] / 2)) / mat[0];
        stImg(0, ii, ij, ik) = fi;
        stImg(1, ii, ij, ik) = fj;
        stImg(2, ii, ij, ik) = fk;
      }
    }
  }

  Log::Print("gewurz", "Foward DFT");
  for (int it = 0; it < nT; it++) {
    thrust::for_each_n(thrust::cuda::par, thrust::make_counting_iterator(0), nS * mat[0] * mat[1] * mat[2],
                       [sKS, stKS, sImg, stImg, it] __device__(int ijks) {
                         int const nI = sImg.extent(0);
                         int const nIJ = nI * sImg.extent(1);
                         int const nIJK = nIJ * sImg.extent(2);
                         int const is = ijks / nIJK;
                         int const k = ijks % nIJK / nIJ;
                         int const j = ijks % nIJK % nIJ / nI;
                         int const i = ijks % nIJK % nIJ % nI;

                         float phase = 0.f;
                         for (int id = 0; id < 3; id++) {
                           phase += stKS(id, is, it) * stImg(id, i, j, k);
                         }
                         auto const ep = cuda::std::exp(cuda::std::complex<float>(0.f, phase));
                         for (int ic = 0; ic < sKS.extent(0); ic++) {
                           sKS(ic, is, it) = ep * sImg(i, j, k, ic);
                         }
                       });
  }

HD5::Writer writer(coreArgs.oname.Get());
writer.writeInfo(info);
}
