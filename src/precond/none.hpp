#pragma once

#include "../types.h"

/* NullOp Preconditioner
 */

template <typename T>
struct NoPrecond
{
  T const &operator()(T const &in) const
  {
    return in;
  }
};
