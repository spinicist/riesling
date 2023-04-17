#pragma once

#include "tensorop.hpp"

namespace rl {

struct GradOp final : TensorOperator<Cx, 4, 5>
{
  OP_INHERIT( Cx, 4, 5 )
  GradOp(InDims const dims);
  OP_DECLARE()
};

struct Grad4Op final : TensorOperator<Cx, 4, 5>
{
  OP_INHERIT( Cx, 4, 5 )
  Grad4Op(InDims const dims);
  OP_DECLARE()
};

} // namespace rl
