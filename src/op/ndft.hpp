#pragma once

#include "basis/basis.hpp"
#include "op/tensorop.hpp"
#include "sdc.hpp"

namespace rl {

template <int NDim>
struct NDFTOp final : TensorOperator<Cx, NDim + 2, 3>
{
  OP_INHERIT(Cx, NDim + 2, 3)
  NDFTOp(Re3 const                             &traj,
         Index const                            nC,
         Sz<NDim> const                         matrix,
         Basis<Cx> const                       &basis = IdBasis(),
         std::shared_ptr<TensorOperator<Cx, 3>> sdc = nullptr);
  OP_DECLARE()

private:
  Re3       traj;
  Re2       xc;
  Index     N, nSamp, nTrace;
  float     scale;
  Basis<Cx> basis;

  std::shared_ptr<TensorOperator<Cx, 3>> sdc;
};

std::shared_ptr<TensorOperator<Cx, 5, 4>> make_ndft(Re3 const                             &traj,
                                                    Index const                            nC,
                                                    Sz3 const                              matrix,
                                                    Basis<Cx> const                       &basis = IdBasis(),
                                                    std::shared_ptr<TensorOperator<Cx, 3>> sdc = nullptr);

} // namespace rl
