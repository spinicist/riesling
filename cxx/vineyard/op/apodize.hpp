#pragma once

#include "kernel/kernel.hpp"
#include "top.hpp"

namespace rl::TOps {

template <typename Scalar_, int AD, int OD = 2> struct Apodize final : TOp<Scalar_, AD + OD, AD + OD>
{
  OP_INHERIT(Cx, AD + OD, AD + OD)
  Apodize(InDims const shape, Sz<AD> const gshape, std::shared_ptr<Kernel<Scalar, AD>> const &k);
  OP_DECLARE(Apodize)

private:
  InDims                res_, brd_;
  Eigen::Tensor<Cx, AD> apo_;
};

} // namespace rl::TOps
