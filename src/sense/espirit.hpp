#pragma once

#include "log.hpp"
#include "op/make_grid.hpp"

namespace rl {
namespace SENSE {
Cx4 ESPIRIT(Cx4 const &grid, Sz3 const shape, Index const kernelRad, Index const calRad, Index const gap, float const thresh);

}
} // namespace rl
