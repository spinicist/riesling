#pragma once

#include "tensorop.hpp"

namespace rl {

struct GradOp final : TensorOperator<Cx, 4, 5>
{
  OP_INHERIT(Cx, 4, 5)
  GradOp(InDims const dims);
  OP_DECLARE()
};

struct GradVecOp final : TensorOperator<Cx, 5, 5>
{
  OP_INHERIT(Cx, 5, 5)
  GradVecOp(InDims const dims);
  OP_DECLARE()
};

} // namespace rl
