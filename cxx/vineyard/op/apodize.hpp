#pragma once

#include "kernel/kernel.hpp"
#include "top.hpp"

namespace rl {

template <typename Scalar_, int NDim>
struct Apodize final : TOp<Scalar_, NDim + 2, NDim + 2>
{
  OP_INHERIT(Cx, NDim + 2, NDim + 2)
  Apodize(InDims const shape, Sz<NDim> const gshape, std::shared_ptr<Kernel<Scalar, NDim>> const &k);
  OP_DECLARE(Apodize)

private:
  InDims                  res_, brd_;
  Eigen::Tensor<Cx, NDim> apo_;
};

} // namespace rl
