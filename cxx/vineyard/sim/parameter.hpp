#pragma once

#include "types.hpp"

namespace rl {

auto ParameterGrid(Index const           nPar,
                   Eigen::ArrayXf const &lo,
                   Eigen::ArrayXf const &hi,
                   Eigen::ArrayXi const &N) -> Eigen::ArrayXXf;

} // namespace rl
