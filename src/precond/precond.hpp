#pragma once

#include "../types.h"

/* NullOp Preconditioner
 */

template <typename T>
struct Precond
{
  virtual T apply(T const &in) const
  {
    return in;
  }

  virtual T inv(T const &in) const
  {
    return in;
  }

  virtual ~Precond(){};
};
