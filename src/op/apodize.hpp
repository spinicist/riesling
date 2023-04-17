#pragma once

#include "tensorop.hpp"
#include "gridBase.hpp"

namespace rl {

template<size_t NDim>
struct ApodizeOp final : TensorOperator<Cx, NDim + 2, NDim + 2>
{
  OP_INHERIT( Cx, NDim + 2, NDim + 2 )
  ApodizeOp(InDims const &inSize, GridBase<Scalar, NDim> *gridder);
  OP_DECLARE()

private:
  InDims res_, brd_;
  Eigen::Tensor<Cx, NDim> apo_;
  void init(GridBase<Scalar, NDim> *gridder);
};

} // namespace rl
