#pragma once

#include "tensorop.hpp"
#include "trajectory.hpp"

#include <memory>
#include <optional>

namespace rl {

// So we can template the kernel size and still stash pointers
template <typename Scalar_, size_t NDim>
struct GridBase : TensorOperator<Scalar_, NDim + 2, 3>
{
  OP_INHERIT(Scalar_, NDim + 2, 3)

  GridBase(InDims const xd, OutDims const yd)
    : Parent(fmt::format("{}D GridOp", NDim), xd, yd)
  {
  }
  virtual ~GridBase(){};

  virtual auto apodization(Sz<NDim> const sz) const -> Eigen::Tensor<float, NDim> = 0;
};

} // namespace rl
