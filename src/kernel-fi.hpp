#pragma once

#include "kernel.hpp"
#include "log.h"
#include "tensorOps.h"

namespace rl {

template <int IP, int TP>
struct FlatIron final : SizedKernel<IP, TP>
{
  using typename SizedKernel<IP, TP>::KTensor;

  FlatIron(float os);
  KTensor k(Point3 const p) const;

private:
  float beta_, scale_;
};

} // namespace rl
