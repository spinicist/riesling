#pragma once

#include "log.hpp"
#include "op/make_grid.hpp"

namespace rl {

Cx4 ESPIRIT(
  GridBase<Cx, 3> *grid, Cx3 data, Index const kernelRad, Index const calRad, Index const gap, float const thresh);

}
