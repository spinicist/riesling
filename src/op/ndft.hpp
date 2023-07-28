#pragma once

#include "basis.hpp"
#include "op/tensorop.hpp"
#include "sdc.hpp"

namespace rl {

template <size_t NDim>
struct NDFTOp final : TensorOperator<Cx, NDim + 2, 3>
{
  OP_INHERIT(Cx, NDim + 2, 3)
  NDFTOp(Re3 const                             &traj,
         Index const                            nC,
         Sz<NDim> const                         matrix,
         Re2 const                             &basis = IdBasis(),
         std::shared_ptr<TensorOperator<Cx, 3>> sdc = nullptr);
  OP_DECLARE()

private:
  Re3 const                              traj;
  Re2 const                              basis;
  std::shared_ptr<TensorOperator<Cx, 3>> sdc;
};

std::shared_ptr<TensorOperator<Cx, 5, 4>> make_ndft(Re3 const                             &traj,
                                                    Index const                            nC,
                                                    Sz3 const                              matrix,
                                                    Re2 const                             &basis = IdBasis(),
                                                    std::shared_ptr<TensorOperator<Cx, 3>> sdc = nullptr);

} // namespace rl