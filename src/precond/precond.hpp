#pragma once

#include "../types.h"

/* NullOp Preconditioner
 */

struct Precond
{
  Cx3 const apply(Cx3 const &in) const
  {
    return in;
  }
};