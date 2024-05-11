#pragma once

#include "basis/basis.hpp"
#include "op/tensorop.hpp"

namespace rl {

template <int NDim>
struct NDFTOp final : TensorOperator<Cx, NDim + 2, 3>
{
  OP_INHERIT(Cx, NDim + 2, 3)
  NDFTOp(Sz<NDim> const matrix, Re3 const &traj, Index const nC, Basis<Cx> const &basis = IdBasis());
  OP_DECLARE(NDFTOp)

  static auto Make(Sz<NDim> const   matrix,
                   Re3 const       &traj,
                   Index const      nC,
                   Basis<Cx> const &basis = IdBasis()) -> std::shared_ptr<NDFTOp<NDim>>;
  void        addOffResonance(Eigen::Tensor<float, NDim> const &f0map, float const t0, float const tSamp);

private:
  Re3       traj;
  Re2       xc;
  Re1       Δf, t;
  Index     N, nSamp, nTrace;
  float     scale;
  Basis<Cx> basis;
};

} // namespace rl
