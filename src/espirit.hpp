#pragma once

#include "log.hpp"
#include "op/make_grid.hpp"

namespace rl {

Cx4 ESPIRIT(
  Cx4 const &grid, Sz3 const outSz, Index const kernelRad, Index const calRad, Index const gap, float const thresh);

}
