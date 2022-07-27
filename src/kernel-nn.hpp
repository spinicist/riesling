#pragma once

#include "kernel.hpp"

namespace rl {

struct NearestNeighbour final : SizedKernel<1, 1>
{
  using typename SizedKernel<1, 1>::KTensor;
  NearestNeighbour();
  KTensor k(Point3 const offset) const;
};

} // namespace rl
