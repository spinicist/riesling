#pragma once

#include "operator.hpp"
#include "gridBase.hpp"

namespace rl {

template<typename Scalar_, size_t NDim>
struct ApodizeOp final : Operator<Scalar_, NDim + 2, NDim + 2>
{
  OP_INHERIT( Scalar_, NDim + 2, NDim + 2 )
  ApodizeOp(InputDims const &inSize, GridBase<Scalar, NDim> *gridder);
  OP_DECLARE()

private:
  InputDims res_, brd_;
  Eigen::Tensor<float, NDim> apo_;
  void init(GridBase<Scalar, NDim> *gridder);
};

} // namespace rl
