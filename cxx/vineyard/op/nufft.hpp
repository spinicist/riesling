#pragma once

#include "op/top.hpp"

#include "apodize.hpp"
#include "grid.hpp"
#include "pad.hpp"

namespace rl::TOps {

template <int NDim> struct NUFFT final : TOp<Cx, NDim + 2, 3>
{
  OP_INHERIT(Cx, NDim + 2, 3)
  NUFFT(Sz<NDim> const           matrix,
        TrajectoryN<NDim> const &traj,
        std::string const       &ktype,
        float const              osamp,
        Index const              nC,
        Basis<Cx> const         &basis = IdBasis<Scalar>(),
        bool const               VCC = false,
        Index const              subgridSz = 32,
        Index const              splitSz = 16384,
        Index const              nBatches = 1);
  OP_DECLARE(NUFFT)

  static auto Make(Sz<NDim> const           matrix,
                   TrajectoryN<NDim> const &traj,
                   GridOpts                &opts,
                   Index const              nC,
                   Basis<Cx> const         &basis) -> std::shared_ptr<NUFFT<NDim>>;

  Grid<Cx, NDim> gridder;
  InTensor mutable workspace;

  TOps::Pad<Cx, NDim + 2, NDim> pad;
  Apodize<Cx, NDim>             apo;
  Index const                   batches;
  Sz<NDim>                      fftDims;
  CxN<NDim>                     fftPh;
};

} // namespace rl::TOps
