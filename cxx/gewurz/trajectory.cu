#include "trajectory.cuh"

#include "rl/log/log.hpp"

namespace gw {

auto ReadTrajectory(rl::HD5::Reader &reader) -> DTensor<TDev, 3>
{
  rl::Log::Print("gewurz", "Read trajectory");
  auto const shape = reader.dimensions("trajectory");
  auto const nS = shape[1];
  auto const nT = shape[2];

  HTensor<float, 3> hT(3L, nS, nT);
  reader.readTo(hT.vec.data(), rl::HD5::Keys::Trajectory);
  HTensor<TDev, 3> hhT(3L, nS, nT);
  thrust::transform(hT.vec.begin(), hT.vec.end(), hhT.vec.begin(), ConvertTo);
  DTensor<TDev, 3> T(3L, nS, nT);
  thrust::copy(hhT.vec.begin(), hhT.vec.end(), T.vec.begin());
  return T;
}

void WriteTrajectory(DTensor<TDev, 3> const &T, rl::HD5::Shape<3> const mat, rl::HD5::Writer &writer)
{
  rl::Log::Print("gewurz", "Write trajectory");
  auto const nS = T.span.extent(1);
  auto const nT = T.span.extent(2);

  HTensor<TDev, 3> hhT(3L, nS, nT);
  thrust::copy(T.vec.begin(), T.vec.end(), hhT.vec.begin());
  HTensor<float, 3> hT(3L, nS, nT);
  thrust::transform(hhT.vec.begin(), hhT.vec.end(), hT.vec.begin(), ConvertFrom);

  writer.writeTensor(rl::HD5::Keys::Trajectory, rl::HD5::Shape<3>(3L, nS, nT), hT.vec.data(), rl::HD5::Dims::Trajectory);
  writer.writeAttribute(rl::HD5::Keys::Trajectory, "matrix", mat);
}

}