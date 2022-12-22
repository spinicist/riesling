#pragma once

#include "operator-alloc.hpp"

namespace rl {

struct GradOp final : OperatorAlloc<Cx, 4, 5>
{
  OPALLOC_INHERIT( Cx, 4, 5 )

  GradOp(InputDims const dims);

  OPALLOC_DECLARE()

private:
};
} // namespace rl
