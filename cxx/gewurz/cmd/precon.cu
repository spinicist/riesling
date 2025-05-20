#define LIBCUDACXX_ENABLE_SIMPLIFIED_COMPLEX_OPERATIONS
#include "rl/io/hd5.hpp"
#include "rl/log/log.hpp"

#include "../algo/lsmr.cuh"
#include "../algo/precon.cuh"
#include "../args.hpp"
#include "../op/dft.cuh"
#include "../op/recon.cuh"
#include "../sense.hpp"
#include "info.hpp"
#include "../trajectory.cuh"
using namespace rl;

void main_precon(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "Input trajectory file");
  args::Positional<std::string> oname(parser, "FILE", "Output weights file");

  ParseCommand(parser, iname, oname);
  Log::Print("Precon", "Welcome!");

  HD5::Reader reader(iname.Get());
  auto const  mat = reader.readAttributeShape<3>(HD5::Keys::Trajectory, "matrix");
  auto const  T = gw::ReadTrajectory(reader);
  auto const  nS = T.span.extent(1);
  auto const  nT = T.span.extent(2);

  auto const W = gw::Preconditioner(T, mat[0], mat[1], mat[2]);

  HTensor<TDev, 2> hW(nS, nT);
  thrust::copy(W.vec.begin(), W.vec.end(), hW.vec.begin());

  HD5::Writer writer(oname.Get());
  writer.writeTensor("weights", HD5::Shape<2>{nS, nT}, hW.vec.data(), {"sample", "trace"});

  Log::Print("Precon", "Finished");
}
