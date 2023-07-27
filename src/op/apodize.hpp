#pragma once

#include "kernel/kernel.hpp"
#include "tensorop.hpp"

namespace rl {

template <typename Scalar_, size_t NDim>
struct ApodizeOp final : TensorOperator<Scalar_, NDim + 2, NDim + 2>
{
  OP_INHERIT(Cx, NDim + 2, NDim + 2)
  ApodizeOp(InDims const shape, Sz<NDim> const gshape, std::shared_ptr<Kernel<Scalar, NDim>> const &k);
  OP_DECLARE()

private:
  InDims                  res_, brd_;
  Eigen::Tensor<Cx, NDim> apo_;
};

} // namespace rl
