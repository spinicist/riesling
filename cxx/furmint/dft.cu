#include "inputs.hpp"

#include "rl/algo/lsmr.hpp"
#include "rl/io/hd5.hpp"
#include "rl/log.hpp"
#include "rl/op/loopify.hpp"
#include "rl/op/ndft.hpp"
#include "rl/sys/threads.hpp"
#include "rl/types.hpp"

using namespace rl;

#define HANDLE_ERROR(x)                                                                                                        \
  {                                                                                                                            \
    const auto err = x;                                                                                                        \
    if (err != CUTENSOR_STATUS_SUCCESS) { throw Log::Failure("CUTENSOR", "Error: {}", cutensorGetErrorString(err)) }           \
  };

#define HANDLE_CUDA_ERROR(x)                                                                                                   \
  {                                                                                                                            \
    const auto err = x;                                                                                                        \
    if (err != cudaSuccess) { throw Log::Failure("CUDA", "Error: {}", cudaGetErrorString(err)); }                              \
  };

void main_ndft(args::Subparser &parser)
{
  CoreArgs coreArgs(parser);
  ParseCommand(parser, coreArgs.iname, coreArgs.oname);

  HD5::Reader reader(coreArgs.iname.Get());
  Info const  info = reader.readInfo();
  auto const  shape = reader.dimensions();
  auto const  nC = shape[0];
  auto const  nSamp = shape[1];
  auto const  nTrace = shape[2];
  HD5::Writer writer(coreArgs.oname.Get());
  writer.writeInfo(info);
  Trajectory traj(reader, info.voxel_size, coreArgs.matrix.Get());
  auto const mat = traj.matrix();
  Cx5 const  noncart = reader.readTensor<Cx5>();
  Cx4 const  images(AddBack(mat, nC));
  Re3 const  ncTraj = traj.points();
  Re4        cTraj(3, mat[0], mat[1], mat[2]);

  for (Index ik = 0; ik < mat[2]; ik++) {
    float const fk = (2.f * M_PI * (ik - mat[2] / 2)) / mat[2];
    for (Index ij = 0; ij < mat[1]; ij++) {
      float const fj = (2.f * M_PI * (ij - mat[1] / 2)) / mat[1];
      for (Index ii = 0; ii < mat[0]; ii++) {
        float const fi = (2.f * M_PI * (ii - mat[0] / 2)) / mat[0];
        cTraj(0, ii, ij, ik) = fi;
        cTraj(1, ii, ij, ik) = fj;
        cTraj(2, ii, ij, ik) = fk;
      }
    }
  }

  std::vector<int> mNCimg{'c', 's', 't'};
  std::vector<int> mCimg{'i', 'j', 'k', 'c'};
  std::vector<int> mE('i','j','k');
  std::vector<int> mEt('s','t');
  std::vector<int> mNCTraj{'d', 's', 't'};
  std::vector<int> mCTraj{'d', 'i', 'j', 'k'};

  std::unordered_map<int, int64_t> xNCimg, xCimg, xNCTraj, xCTraj, xE;
  xNCimg['c'] = nC;
  xNCimg['s'] = nSamp;
  xNCimg['t'] = nTrace;
  xCimg['i'] = mat[0];
  xCimg['j'] = mat[1];
  xCimg['k'] = mat[2];
  xCimg['c'] = nC;
  xNCTraj['d'] = 3;
  xNCTraj['s'] = nSamp;
  xNCTraj['t'] = nTrace;
  xCTraj['d'] = 3;
  xCTraj['i'] = mat[0];
  xCTraj['j'] = mat[1];
  xCTraj['k'] = mat[2];
  xE['i'] = mat[0];
  xE['j'] = mat[1];
  xE['k'] = mat[2];
  xEt['s'] = nSamp;
  xEt['t'] = nTrace;

  void *gpuNCimg, *gpuCimg, *gpuNCTraj, *gpuCTraj;

  HANDLE_CUDA_ERROR(cudaMalloc((void **)&gpuNCimg, mNCimg.size() * sizeof(Cx)));
  HANDLE_CUDA_ERROR(cudaMalloc((void **)&gpuCimg, mCimg.size() * sizeof(Cx)));
  HANDLE_CUDA_ERROR(cudaMalloc((void **)&gpuNCTraj, ncTraj.size() * sizeof(Re)));
  HANDLE_CUDA_ERROR(cudaMalloc((void **)&gpuCTraj, cTraj.size() * sizeof(Re)));

  HANDLE_CUDA_ERROR(cudaMemcpy(gpuNCimg, noncart.data(), cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(gpuNCTraj, ncTraj.data(), cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(gpuCTraj, cTraj.data(), cudaMemcpyHostToDevice));

  cutensorHandle_t handle;
  HANDLE_ERROR(cutensorCreate(&handle));

  cutensorTensorDescriptor_t dNCimg, dCimg, dNCTraj, dCTraj;
  uint32_t const kAlignment = 128;
  HANDLE_ERROR(cutensorCreateTensorDescriptor(handle, &dNCimg, mNCimg, xNCimg.data(), NULL, CUTENSOR_C_32F, kAlignment));
  HANDLE_ERROR(cutensorCreateTensorDescriptor(handle, &dCimg, mCimg, xCimg.data(), NULL, CUTENSOR_C_32F, kAlignment));
  HANDLE_ERROR(cutensorCreateTensorDescriptor(handle, &dNCTraj, mNCTraj, xNCTraj.data(), NULL, CUTENSOR_R_32F, kAlignment));
  HANDLE_ERROR(cutensorCreateTensorDescriptor(handle, &dCTraj, mCTraj, xCTraj.data(), NULL, CUTENSOR_R_32F, kAlignment));

  writer.writeTensor(HD5::Keys::Data, cart.dimensions(), cart.data(), HD5::Dims::Channels);
}
