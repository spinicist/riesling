#pragma once

#include "operator-alloc.hpp"

namespace rl {

struct GradOp final : OperatorAlloc<Cx, 4, 5>
{
  OPALLOC_INHERIT( Cx, 4, 5 )
  GradOp(InputDims const dims);
  OPALLOC_DECLARE()
};

struct Grad4Op final : OperatorAlloc<Cx, 4, 5>
{
  OPALLOC_INHERIT( Cx, 4, 5 )
  Grad4Op(InputDims const dims);
  OPALLOC_DECLARE()
};

} // namespace rl
