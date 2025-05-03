#pragma once

#include "lsmr.cuh"
#include "types.cuh"

namespace gw {

namespace DFT {

struct ThreeD : Op<CuCxF, 3, 2>
{
  using XT = DTensor<CuCxF, 3>::Span;
  using YT = DTensor<CuCxF, 2>::Span;
  using TT = DTensor<float, 3>::Span;
  TT traj;
  ThreeD(TT t) : traj(t) {};
  void forward(XT x, YT y) const override;
  void adjoint(YT y, XT x) const override;
};

struct ThreeDPacked : Op<CuCxF, 4, 3>
{
  using XT = DTensor<CuCxF, 4>::Span;
  using YT = DTensor<CuCxF, 3>::Span;
  using TT = DTensor<float, 3>::Span;
  TT traj;
  ThreeDPacked(TT t) : traj(t) {};
  void forward(XT x, YT y) const override;
  void adjoint(YT y, XT x) const override;
};

} // namespace DFT

} // namespace gw
