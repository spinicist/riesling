#pragma once

#include "types.hpp"

namespace rl {

namespace Parameters {
auto T1(Index const nS) -> Eigen::ArrayXXf;
auto T1T2(Index const nS) -> Eigen::ArrayXXf;
auto T1η(Index const nS) -> Eigen::ArrayXXf;
}

} // namespace rl
