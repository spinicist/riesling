#pragma once

#include "lsmr.cuh"
#include "types.cuh"

#include <cuda/experimental/stream.cuh>

namespace cudax = cuda::experimental;

namespace gw {

namespace DFT {

struct ThreeD2 : Op<CuCx<TDev>, 3, 2>
{
  using XT = DTensor<CuCx<TDev>, 3>::Span;
  using YT = DTensor<CuCx<TDev>, 2>::Span;
  using TT = DTensor<TDev, 3>::Span;
  TT traj;
  cudax::stream stream;
  ThreeD2(TT t)
    : traj(t) {};
  void forward(XT x, YT y) const override;
  void adjoint(YT y, XT x) const override;
};

template <int NP> struct ThreeDPacked2 : Op<CuCx<TDev>, 4, 3>
{
  using XT = DTensor<CuCx<TDev>, 4>::Span;
  using YT = DTensor<CuCx<TDev>, 3>::Span;
  using TT = DTensor<TDev, 3>::Span;
  TT traj;
  cudax::stream stream;
  ThreeDPacked2(TT t)
    : traj(t) {};
  void forward(XT x, YT y) const override;
  void adjoint(YT y, XT x) const override;
};

} // namespace DFT

} // namespace gw
