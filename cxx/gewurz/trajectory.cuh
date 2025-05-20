#pragma once

#include "rl/io/hd5.hpp"
#include "types.cuh"

namespace gw {

auto ReadTrajectory(rl::HD5::Reader &reader) -> DTensor<TDev, 3>;
void WriteTrajectory(DTensor<TDev, 3> const &T, rl::HD5::Shape<3> const mat, rl::HD5::Writer &writer);

}