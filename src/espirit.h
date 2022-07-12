#pragma once

#include "log.h"
#include "op/gridBase.hpp"

namespace rl {

Cx4 ESPIRIT(
  GridBase<Cx> *grid, Cx3 const &data, Index const kernelRad, Index const calRad, Index const gap, float const thresh);

}
