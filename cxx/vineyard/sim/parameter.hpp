#pragma once

#include "types.hpp"

namespace rl {

auto ParameterGrid(Index const           nPar,
                   Eigen::ArrayXf const &lo,
                   Eigen::ArrayXf const &hi,
                   Eigen::ArrayXf const &delta) -> Eigen::ArrayXXf;

} // namespace rl
