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
  Eigen::Matrix<float, NDim, -1>         traj;
  Eigen::ArrayXXcf                       basis;
  std::shared_ptr<TensorOperator<Cx, 3>> sdc;
  Index                                  N, nSamp, nTrace;
  float                                  scale;
  Eigen::Matrix<float, NDim, -1>         xc;
};

std::shared_ptr<TensorOperator<Cx, 5, 4>> make_ndft(Re3 const                             &traj,
                                                    Index const                            nC,
                                                    Sz3 const                              matrix,
                                                    Basis<Cx> const                       &basis = IdBasis(),
                                                    std::shared_ptr<TensorOperator<Cx, 3>> sdc = nullptr);

} // namespace rl
