#include "kernel-nn.hpp"
#include "log.h"

namespace rl {

NearestNeighbour::NearestNeighbour()
{
  Log::Print("Nearest-neighbour kernel");
}

auto NearestNeighbour::k(Point3 const offset) const -> KTensor
{
  KTensor k;
  k.setConstant(1.f);
  return k;
}

} // namespace rl