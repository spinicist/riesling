#pragma once

#include "log/log.hpp"

namespace rl {
namespace SENSE {
Cx4 ESPIRIT(Cx4 const &grid, Sz3 const shape, Index const kernelRad, Index const calRad, Index const gap, float const thresh);

}
} // namespace rl
