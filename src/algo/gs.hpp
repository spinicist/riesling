#pragma once

#include "types.hpp"

namespace rl {

auto GramSchmidt(Eigen::MatrixXcf const &mat) -> Eigen::MatrixXcf;

}
