#pragma once

#include "op.cuh"
#include "types.cuh"

namespace gw {

template <int NC> struct Recon : Op<CuCx<TDev>, 3, 3>
{
  using XT = DTensor<CuCx<TDev>, 3>::Span;
  using YT = DTensor<CuCx<TDev>, 3>::Span;
  using ST = DTensor<CuCx<TDev>, 4>::Span;
  using TT = DTensor<TDev, 3>::Span;
  ST const sense;
  TT const traj;

  Recon(ST s, TT t);
  void forward(XT x, YT y) const override;
  void adjoint(YT y, XT x) const override;
};

} // namespace gw
