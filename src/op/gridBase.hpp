#pragma once

#include "operator-alloc.hpp"
#include "trajectory.hpp"

#include <memory>
#include <optional>

namespace rl {

// So we can template the kernel size and still stash pointers
template <typename Scalar_, size_t NDim>
struct GridBase : OperatorAlloc<Scalar_, NDim + 2, 3>
{
  OPALLOC_INHERIT(Scalar_, NDim + 2, 3)

  GridBase(InputDims const xd, OutputDims const yd)
    : Parent(fmt::format(FMT_STRING("{}D GridOp"), NDim), xd, yd)
  {
  }
  virtual ~GridBase(){};

  virtual auto apodization(Sz<NDim> const sz) const -> Eigen::Tensor<float, NDim> = 0;
};

} // namespace rl
